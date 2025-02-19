#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksIterative_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksIterative_h

#include "KokkosCore/hintLightWeight.h"
#include "KokkosCore/HistoContainer.h"
#include "KokkosCore/atomic.h"
#include "KokkosCore/kokkos_assert.h"

#include "gpuVertexFinder.h"
#include "gpuClusterFillHist.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    template <typename Hist>
    KOKKOS_INLINE_FUNCTION void clusterTracksIterative(
        const Kokkos::View<ZVertices, KokkosExecSpace, Restrict>& vdata,
        const Kokkos::View<WorkSpace, KokkosExecSpace, Restrict>& vws,
        int minT,       // min number of neighbours to be "seed"
        float eps,      // max absolute distance to cluster
        float errmax,   // max error to be "seed"
        float chi2max,  // max normalized distance to cluster
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      Hist* hist = static_cast<Hist*>(team_member.team_shmem().get_shmem(sizeof(Hist)));
      Kokkos::View<Hist, KokkosExecSpace::scratch_memory_space, RestrictUnmanaged> histView(hist);

      clusterFillHist(vdata, vws, histView, minT, eps, errmax, chi2max, team_member);
      team_member.team_barrier();

      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      const auto leagueRank = team_member.league_rank();
      const auto teamRank = team_member.team_rank();
      const auto teamSize = team_member.team_size();
      const auto id = leagueRank * teamSize + teamRank;

      auto er2mx = errmax * errmax;

      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();

      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;
      float const* __restrict__ ezt2 = ws.ezt2;

      uint32_t& nvFinal = data.nvFinal;
      uint32_t& nvIntermediate = ws.nvIntermediate;

      uint8_t* __restrict__ izt = ws.izt;
      int32_t* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

      assert(hist->size() == nt);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) { hist->fill(izt[i], uint16_t(i)); });
      team_member.team_barrier();

      // count neighbours
      // TODO: can't use parallel_for+TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (ezt2[i] > er2mx)
          continue;
        auto loop = [&](uint32_t j) {
          if (i == j)
            return;
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > eps)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          nn[i]++;
        };

        forEachInBins(hist, izt[i], 1, loop);
      }

      int* nloops = static_cast<int*>(team_member.team_shmem().get_shmem(sizeof(int)));
      nloops[0] = 0;

      team_member.team_barrier();

      // cluster seeds only
      int more = 1;
      while (more) {
        if (1 == nloops[0] % 2) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
            auto m = iv[i];
            while (m != iv[m])
              m = iv[m];
            iv[i] = m;
          });
        } else {
          more = 0;
          // TODO: can't use parallel_for+TeamThreadRange because of "continue"
          for (unsigned k = teamRank; k < hist->size(); k += teamSize) {
            auto p = hist->begin() + k;
            auto i = (*p);
            auto be = std::min(Hist::bin(izt[i]) + 1, int(hist->nbins() - 1));
            if (nn[i] < minT)
              continue;  // DBSCAN core rule
            auto loop = [&](uint32_t j) {
              assert(i != j);
              if (nn[j] < minT)
                return;  // DBSCAN core rule
              auto dist = std::abs(zt[i] - zt[j]);
              if (dist > eps)
                return;
              if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
                return;
              auto old = cms::kokkos::atomic_fetch_min(&iv[j], iv[i]);
              if (old != iv[i]) {
                // end the loop only if no changes were applied
                more = 1;
              }
              cms::kokkos::atomic_min_fetch(&iv[i], old);
            };
            ++p;
            for (; p < hist->end(be); ++p)
              loop(*p);
          }  // for i
        }
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() { ++nloops[0]; });

        team_member.team_reduce(Kokkos::Sum<int>(more));
      }  // while

      // collect edges (assign to closest cluster of closest point??? here to closest point)
      // TODO: can't use parallel_for+TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
        if (nn[i] >= minT)
          continue;  // DBSCAN edge rule
        float mdist = eps;
        auto loop = [&](int j) {
          if (nn[j] < minT)
            return;  // DBSCAN core rule
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;  // needed?
          mdist = dist;
          iv[i] = iv[j];  // assign to cluster (better be unique??)
        };
        forEachInBins(hist, izt[i], 1, loop);
      }

      unsigned int* foundClusters =
          static_cast<unsigned int*>(team_member.team_shmem().get_shmem(sizeof(unsigned int)));
      foundClusters[0] = 0;
      team_member.team_barrier();

      // find the number of different clusters, identified by a tracks with clus[i] == i;
      // mark these tracks with a negative id.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        if (iv[i] == int(i)) {
          if (nn[i] >= minT) {
            auto old = cms::kokkos::atomic_fetch_add(foundClusters, 1U);
            iv[i] = -(old + 1);
          } else {  // noise
            iv[i] = -9998;
          }
        }
      });
      team_member.team_barrier();

      assert(foundClusters[0] < ZVertices::MAXVTX);

      // propagate the negative id to all the tracks in the cluster.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        if (iv[i] >= 0) {
          // mark each track in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
      });
      team_member.team_barrier();

      // adjust the cluster id to be a positive value starting from 0
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) { iv[i] = -iv[i] - 1; });

      nvIntermediate = nvFinal = foundClusters[0];

      if (verbose && 0 == id)
        printf("found %d proto vertices\n", foundClusters[0]);
    }

    template <typename ExecSpace>
    void clusterTracksIterativeHost(const Kokkos::View<ZVertices, ExecSpace, Restrict>& vdata,
                                    const Kokkos::View<WorkSpace, ExecSpace, Restrict>& vws,
                                    int minT,       // min number of neighbours to be "seed"
                                    float eps,      // max absolute distance to cluster
                                    float errmax,   // max error to be "seed"
                                    float chi2max,  // max normalized distance to cluster
                                    const ExecSpace& execSpace,
                                    Kokkos::TeamPolicy<ExecSpace> policy) {
      using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

      using Hist = cms::kokkos::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
      // TODO: don't really understand why 8 additional bytes are needed. Some internal bookkeeping?
      auto shared_mem_bytes = sizeof(Hist) + 8 + sizeof(unsigned int) + 8 + sizeof(int);

      Kokkos::parallel_for(
          "clusterTracksIterative",
          hintLightWeight(policy.set_scratch_size(0, Kokkos::PerTeam(shared_mem_bytes))),
          KOKKOS_LAMBDA(const member_type& team_member) {
            clusterTracksIterative<Hist>(vdata, vws, minT, eps, errmax, chi2max, team_member);
          });
    }
  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksIterative_h
