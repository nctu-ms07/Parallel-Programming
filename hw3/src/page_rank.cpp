#include "page_rank.h"

#include <cmath>

#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  #pragma omp parallel for
  for (int i = 0; i < numNodes; i++) {
    solution[i] = equal_prob;
  }

  /*
   For PP students: Implement the page rank algorithm here.  You
   are expected to parallelize the algorithm using openMP.  Your
   solution may need to allocate (and free) temporary arrays.

   Basic page rank pseudocode is provided below to get you started:

   // initialization: see example code above
   score_old[vi] = 1/numNodes;

   while (!converged) {

     // compute score_new[vi] for all nodes vi:
     score_new[vi] = sum over all nodes vj reachable from incoming edges
                        { score_old[vj] / number of edges leaving vj  }
     score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

     score_new[vi] += sum over all nodes v in graph with no outgoing edges
                        { damping * score_old[v] / numNodes }

     // compute how much per-node scores have changed
     // quit once algorithm has converged

     global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
     converged = (global_diff < convergence)
   }

   */

  auto *const k_double_ptr = (double *)malloc(numNodes * sizeof(double));

  double *score_new, *score_old;
  score_new = solution;
  score_old = k_double_ptr;

  double global_diff = convergence;
  double no_outgoing_sum;

  while (global_diff >= convergence) {
    global_diff = 0.0;
    no_outgoing_sum = 0.0;

    std::swap(score_old, score_new);

    #pragma omp parallel
    {

      #pragma omp for reduction (+ : no_outgoing_sum)
      for (int vi = 0; vi < numNodes; vi++) {
        if (outgoing_size(g, vi) == 0) {
          no_outgoing_sum += damping * score_old[vi] / numNodes;
        }
      }

      #pragma omp for reduction (+ : global_diff)
      for (int vi = 0; vi < numNodes; vi++) {
        const Vertex *start = incoming_begin(g, vi);
        const Vertex *end = incoming_end(g, vi);
        double score_sum = 0.0;
        for (const Vertex *incoming = start; incoming != end; incoming++) {
          score_sum += score_old[*incoming] / outgoing_size(g, *incoming);
        }
        score_new[vi] = (damping * score_sum) + (1.0 - damping) / numNodes + no_outgoing_sum;

        global_diff += fabs(score_new[vi] - score_old[vi]);
      }
    }
  }

  if(score_new != solution){
    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
      solution[i] = score_new[i];
    }
  }

  free(k_double_ptr);
}
