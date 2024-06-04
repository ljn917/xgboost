/*!
 * Copyright 2015-2022 XGBoost contributors
 */
#include <dmlc/omp.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>

#include "xgboost/json.h"
#include "xgboost/parameter.h"

#include "../common/math.h"
#include "../common/random.h"

namespace xgboost {
namespace common {
// comparator functions for sorting pairs in descending order
inline static bool CmpFirst(const std::pair<float, unsigned> &a,
                            const std::pair<float, unsigned> &b) {
  return a.first > b.first;
}
}

namespace obj {

#if defined(XGBOOST_USE_CUDA) && !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(rank_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct LambdaRankParam : public XGBoostParameter<LambdaRankParam> {
  size_t num_pairsample;
  float fix_list_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LambdaRankParam) {
    DMLC_DECLARE_FIELD(num_pairsample).set_lower_bound(1).set_default(1)
        .describe("Number of pair generated for each instance.");
    DMLC_DECLARE_FIELD(fix_list_weight).set_lower_bound(0.0f).set_default(0.0f)
        .describe("Normalize the weight of each list by this value,"
                  " if equals 0, no effect will happen");
  }
};

/*! \brief helper information in a list */
struct ListEntry {
  /*! \brief the predict score we in the data */
  bst_float pred;
  /*! \brief the actual label of the entry */
  bst_float label;
  /*! \brief row index in the data matrix */
  unsigned rindex;
  // constructor
  ListEntry(bst_float pred, bst_float label, unsigned rindex)
    : pred(pred), label(label), rindex(rindex) {}
  // comparator by prediction
  inline static bool CmpPred(const ListEntry &a, const ListEntry &b) {
    return a.pred > b.pred;
  }
  // comparator by label
  inline static bool CmpLabel(const ListEntry &a, const ListEntry &b) {
    return a.label > b.label;
  }
};

/*! \brief a pair in the lambda rank */
struct LambdaPair {
  /*! \brief positive index: this is a position in the list */
  unsigned pos_index;
  /*! \brief negative index: this is a position in the list */
  unsigned neg_index;
  /*! \brief weight to be filled in */
  bst_float weight;
  // constructor
  LambdaPair(unsigned pos_index, unsigned neg_index)
    : pos_index(pos_index), neg_index(neg_index), weight(1.0f) {}
  // constructor
  LambdaPair(unsigned pos_index, unsigned neg_index, bst_float weight)
    : pos_index(pos_index), neg_index(neg_index), weight(weight) {}
};

class PairwiseLambdaWeightComputer {
 public:
  /*!
   * \brief get lambda weight for existing pairs - for pairwise objective
   * \param list a list that is sorted by pred score
   * \param io_pairs record of pairs, containing the pairs to fill in weights
   */
  static void GetLambdaWeight(const std::vector<ListEntry>&,
                              std::vector<LambdaPair>*) {}

  static char const* Name() {
    return "rank:pairwise-v1";
  }
};

class MAPLambdaWeightComputer
{
 public:
  struct MAPStats {
    /*! \brief the accumulated precision */
    float ap_acc{0.0f};
    /*!
     * \brief the accumulated precision,
     *   assuming a positive instance is missing
     */
    float ap_acc_miss{0.0f};
    /*!
     * \brief the accumulated precision,
     * assuming that one more positive instance is inserted ahead
     */
    float ap_acc_add{0.0f};
    /* \brief the accumulated positive instance count */
    float hits{0.0f};

    XGBOOST_DEVICE MAPStats() {}  // NOLINT
    XGBOOST_DEVICE MAPStats(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits)
      : ap_acc(ap_acc), ap_acc_miss(ap_acc_miss), ap_acc_add(ap_acc_add), hits(hits) {}

    // For prefix scan
    XGBOOST_DEVICE MAPStats operator +(const MAPStats &v1) const {
      return {ap_acc + v1.ap_acc, ap_acc_miss + v1.ap_acc_miss,
              ap_acc_add + v1.ap_acc_add, hits + v1.hits};
    }

    // For test purposes - compare for equality
    XGBOOST_DEVICE bool operator ==(const MAPStats &rhs) const {
      return ap_acc == rhs.ap_acc && ap_acc_miss == rhs.ap_acc_miss &&
             ap_acc_add == rhs.ap_acc_add && hits == rhs.hits;
    }
  };

 private:
  template <typename T>
  XGBOOST_DEVICE inline static void Swap(T &v0, T &v1) {
    std::swap(v0, v1);
  }

  /*!
   * \brief Obtain the delta MAP by trying to switch the positions of labels in pos_pred_pos or
   *        neg_pred_pos when sorted by predictions
   * \param pos_pred_pos positive label's prediction value position when the groups prediction
   *        values are sorted
   * \param neg_pred_pos negative label's prediction value position when the groups prediction
   *        values are sorted
   * \param pos_label, neg_label the chosen positive and negative labels
   * \param p_map_stats a vector containing the accumulated precisions for each position in a list
   * \param map_stats_size size of the accumulated precisions vector
   */
  XGBOOST_DEVICE inline static bst_float GetLambdaMAP(
    int pos_pred_pos, int neg_pred_pos,
    bst_float pos_label, bst_float neg_label,
    const MAPStats *p_map_stats, uint32_t map_stats_size) {
    if (pos_pred_pos == neg_pred_pos || p_map_stats[map_stats_size - 1].hits == 0) {
      return 0.0f;
    }
    if (pos_pred_pos > neg_pred_pos) {
      Swap(pos_pred_pos, neg_pred_pos);
      Swap(pos_label, neg_label);
    }
    bst_float original = p_map_stats[neg_pred_pos].ap_acc;
    if (pos_pred_pos != 0) original -= p_map_stats[pos_pred_pos - 1].ap_acc;
    bst_float changed = 0;
    bst_float label1 = pos_label > 0.0f ? 1.0f : 0.0f;
    bst_float label2 = neg_label > 0.0f ? 1.0f : 0.0f;
    if (label1 == label2) {
      return 0.0;
    } else if (label1 < label2) {
      changed += p_map_stats[neg_pred_pos - 1].ap_acc_add - p_map_stats[pos_pred_pos].ap_acc_add;
      changed += (p_map_stats[pos_pred_pos].hits + 1.0f) / (pos_pred_pos + 1);
    } else {
      changed += p_map_stats[neg_pred_pos - 1].ap_acc_miss - p_map_stats[pos_pred_pos].ap_acc_miss;
      changed += p_map_stats[neg_pred_pos].hits / (neg_pred_pos + 1);
    }
    bst_float ans = (changed - original) / (p_map_stats[map_stats_size - 1].hits);
    if (ans < 0) ans = -ans;
    return ans;
  }

 public:
  /*
   * \brief obtain preprocessing results for calculating delta MAP
   * \param sorted_list the list containing entry information
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline static void GetMAPStats(const std::vector<ListEntry> &sorted_list,
                                 std::vector<MAPStats> *p_map_acc) {
    std::vector<MAPStats> &map_acc = *p_map_acc;
    map_acc.resize(sorted_list.size());
    bst_float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (size_t i = 1; i <= sorted_list.size(); ++i) {
      if (sorted_list[i - 1].label > 0.0f) {
        hit++;
        acc1 += hit / i;
        acc2 += (hit - 1) / i;
        acc3 += (hit + 1) / i;
      }
      map_acc[i - 1] = MAPStats(acc1, acc2, acc3, hit);
    }
  }

  static char const* Name() {
    return "rank:map-v1";
  }

  static void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                              std::vector<LambdaPair> *io_pairs) {
    std::vector<LambdaPair> &pairs = *io_pairs;
    std::vector<MAPStats> map_stats;
    GetMAPStats(sorted_list, &map_stats);
    for (auto & pair : pairs) {
      pair.weight *=
        GetLambdaMAP(pair.pos_index, pair.neg_index,
                     sorted_list[pair.pos_index].label, sorted_list[pair.neg_index].label,
                     &map_stats[0], map_stats.size());
    }
  }
};

// objective for lambda rank
template <typename LambdaWeightComputerT>
class LambdaRankObj : public ObjFunction {
 public:
  void Configure(Args const &args) override { param_.UpdateAllowUnknown(args); }
  ObjInfo Task() const override { return ObjInfo::kRanking; }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CHECK_EQ(preds.Size(), info.labels.Size()) << "label size predict size not match";

    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(info.labels.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK(gptr.size() != 0 && gptr.back() == info.labels.Size())
          << "group structure not consistent with #rows" << ", "
          << "group ponter size: " << gptr.size() << ", "
          << "labels size: " << info.labels.Size() << ", "
          << "group pointer back: " << (gptr.size() == 0 ? 0 : gptr.back());

      ComputeGradientsOnCPU(preds, info, iter, out_gpair, gptr);
  }

  const char* DefaultEvalMetric() const override {
    return "map";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(LambdaWeightComputerT::Name());
    out["lambda_rank_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["lambda_rank_param"], &param_);
  }

 private:
  bst_float ComputeWeightNormalizationFactor(const MetaInfo& info,
                                             const std::vector<unsigned> &gptr) {
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    bst_float sum_weights = 0;
    for (bst_omp_uint k = 0; k < ngroup; ++k) {
      sum_weights += info.GetWeight(k);
    }
    return ngroup / sum_weights;
  }

  void ComputeGradientsOnCPU(const HostDeviceVector<bst_float>& preds,
                             const MetaInfo& info,
                             std::int32_t iter,
                             linalg::Matrix<GradientPair>* out_gpair,
                             const std::vector<unsigned> &gptr) {
    LOG(DEBUG) << "Computing " << LambdaWeightComputerT::Name() << " gradients on CPU.";

    bst_float weight_normalization_factor = ComputeWeightNormalizationFactor(info, gptr);

    const auto& preds_h = preds.HostVector();
    const auto& labels = info.labels.HostView();
    out_gpair->Reshape(preds.Size(), 1);
    auto gpair = out_gpair->HostView();
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);

    dmlc::OMPException exc;
#pragma omp parallel num_threads(ctx_->Threads())
    {
      exc.Run([&]() {
        // parallel construct, declare random number generator here, so that each
        // thread use its own random number generator, seed by thread id and current iteration
        std::minstd_rand rnd((iter + 1) * 1111);
        std::vector<LambdaPair> pairs;
        std::vector<ListEntry>  lst;
        std::vector< std::pair<bst_float, unsigned> > rec;

        #pragma omp for schedule(static)
        for (bst_omp_uint k = 0; k < ngroup; ++k) {
          exc.Run([&]() {
            lst.clear(); pairs.clear();
            for (unsigned j = gptr[k]; j < gptr[k+1]; ++j) {
              lst.emplace_back(preds_h[j], labels(j), j);
              gpair(j) = GradientPair(0.0f, 0.0f);
            }
            std::stable_sort(lst.begin(), lst.end(), ListEntry::CmpPred);
            rec.resize(lst.size());
            for (unsigned i = 0; i < lst.size(); ++i) {
              rec[i] = std::make_pair(lst[i].label, i);
            }
            std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
            // enumerate buckets with same label
            // for each item in the lst, grab another sample randomly
            for (unsigned i = 0; i < rec.size(); ) {
              unsigned j = i + 1;
              while (j < rec.size() && rec[j].first == rec[i].first) ++j;
              // bucket in [i,j), get a sample outside bucket
              unsigned nleft = i, nright = static_cast<unsigned>(rec.size() - j);
              if (nleft + nright != 0) {
                int nsample = param_.num_pairsample;
                while (nsample --) {
                  for (unsigned pid = i; pid < j; ++pid) {
                    unsigned ridx =
                        std::uniform_int_distribution<unsigned>(0, nleft + nright - 1)(rnd);
                    if (ridx < nleft) {
                      pairs.emplace_back(rec[ridx].second, rec[pid].second,
                          info.GetWeight(k) * weight_normalization_factor);
                    } else {
                      pairs.emplace_back(rec[pid].second, rec[ridx+j-i].second,
                          info.GetWeight(k) * weight_normalization_factor);
                    }
                  }
                }
              }
              i = j;
            }
            // get lambda weight for the pairs
            LambdaWeightComputerT::GetLambdaWeight(lst, &pairs);
            // rescale each gradient and hessian so that the lst have constant weighted
            float scale = 1.0f / param_.num_pairsample;
            if (param_.fix_list_weight != 0.0f) {
              scale *= param_.fix_list_weight / (gptr[k + 1] - gptr[k]);
            }
            for (auto & pair : pairs) {
              const ListEntry &pos = lst[pair.pos_index];
              const ListEntry &neg = lst[pair.neg_index];
              const bst_float w = pair.weight * scale;
              const float eps = 1e-16f;
              bst_float p = common::Sigmoid(pos.pred - neg.pred);
              bst_float g = p - 1.0f;
              bst_float h = std::max(p * (1.0f - p), eps);
              // accumulate gradient and hessian in both pid, and nid
              gpair(pos.rindex) += GradientPair(g * w, 2.0f*w*h);
              gpair(neg.rindex) += GradientPair(-g * w, 2.0f*w*h);
            }
          });
        }
      });
    }
    exc.Rethrow();
  }

  LambdaRankParam param_;
};

#if !defined(GTEST_TEST)
// register the objective functions
DMLC_REGISTER_PARAMETER(LambdaRankParam);

XGBOOST_REGISTER_OBJECTIVE(PairwiseRankObj, PairwiseLambdaWeightComputer::Name())
.describe("Pairwise rank objective.")
.set_body([]() { return new LambdaRankObj<PairwiseLambdaWeightComputer>(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankObjMAP, MAPLambdaWeightComputer::Name())
.describe("LambdaRank with MAP as objective.")
.set_body([]() { return new LambdaRankObj<MAPLambdaWeightComputer>(); });
#endif

}  // namespace obj
}  // namespace xgboost
