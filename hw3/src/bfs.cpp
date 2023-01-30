#include "bfs.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_DISTANCE -1
#define THRESHOLD 0.25

void vertex_set_clear(vertex_set *list) {
  list->count = 0;
}

void vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *) malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

int top_down_step(Graph g, int *const distances, const int &travel_distance) {
  int new_frontiers_size = 0;

  #pragma omp parallel for reduction (+ : new_frontiers_size) schedule (dynamic, 1024)
  for (int vi = 0; vi < g->num_nodes; vi++) {
    // frontier vertex's distance equals to travel distance
    if (distances[vi] == travel_distance) {
      const int start_edge = g->outgoing_starts[vi];
      const int end_edge = (vi == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[vi + 1];

      for (int edge = start_edge; edge < end_edge; edge++) {
        const int out_going = g->outgoing_edges[edge];
        // update distance of every outgoing-neighbor vertex that's not been visited to (travel distance + 1)
        if (distances[out_going] == NOT_VISITED_DISTANCE) {
          distances[out_going] = travel_distance + 1;
          new_frontiers_size += 1;
        }
      }
    }
  }

  return new_frontiers_size;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

  // initialize distances to NOT_VISITED_DISTANCE
  memset(sol->distances, NOT_VISITED_DISTANCE, graph->num_nodes * sizeof(int));

  // distance to root node;
  int travel_distance = 0;
  // setup distance of ROOT_NODE_ID
  sol->distances[ROOT_NODE_ID] = travel_distance;
  int frontier_size = 1;

  while (frontier_size > 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d ", frontier_size);
#endif

    frontier_size = top_down_step(graph, sol->distances, travel_distance);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("%.4f sec\n", end_time - start_time);
#endif

    travel_distance += 1;
  }
}

int bottom_up_step(Graph g, int *distances, const int &travel_distance) {
  int new_frontiers_size = 0;

  #pragma omp parallel for reduction (+ : new_frontiers_size) schedule (dynamic, 1024)
  for (int vi = 0; vi < g->num_nodes; vi++) {
    if (distances[vi] == NOT_VISITED_DISTANCE) {
      const int start_edge = g->incoming_starts[vi];
      const int end_edge = (vi == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[vi + 1];

      for (int edge = start_edge; edge < end_edge; edge++) {
        const int in_coming = g->incoming_edges[edge];
        // frontier vertex's distance equals to travel distance
        if (distances[in_coming] == travel_distance) {
          // update distance of current vertex vi to (travel distance + 1)
          distances[vi] = travel_distance + 1;
          new_frontiers_size += 1;
          break;
        }
      }
    }
  }

  return new_frontiers_size;

}

void bfs_bottom_up(Graph graph, solution *sol) {
  // For PP students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.

  // initialize distances to NOT_VISITED_DISTANCE
  memset(sol->distances, NOT_VISITED_DISTANCE, graph->num_nodes * sizeof(int));

  // distance to root node;
  int travel_distance = 0;
  // setup distance of ROOT_NODE_ID
  sol->distances[ROOT_NODE_ID] = travel_distance;
  int frontier_size = 1;

  while (frontier_size > 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d ", frontier_size);
#endif

    frontier_size = bottom_up_step(graph, sol->distances, travel_distance);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("%.4f sec\n", end_time - start_time);
#endif

    travel_distance += 1;
  }
}

void bfs_hybrid(Graph graph, solution *sol) {
  // For PP students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.

  // initialize distances to NOT_VISITED_DISTANCE
  memset(sol->distances, NOT_VISITED_DISTANCE, graph->num_nodes * sizeof(int));

  // distance to root node;
  int travel_distance = 0;
  // setup distance of ROOT_NODE_ID
  sol->distances[ROOT_NODE_ID] = travel_distance;
  int frontier_size = 1;

  const double kThreshold_size = graph->num_nodes * THRESHOLD;

  while (frontier_size > 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d ", frontier_size);
#endif

    if (frontier_size < kThreshold_size) {
      frontier_size = top_down_step(graph, sol->distances, travel_distance);
    } else {
      frontier_size = bottom_up_step(graph, sol->distances, travel_distance);
    }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("%.4f sec\n", end_time - start_time);
#endif

    travel_distance += 1;
  }
}

