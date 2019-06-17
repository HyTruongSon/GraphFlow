#include "../GraphFlow/DenseGraph.h"
#include <vector>
#include <string>
#include <tuple>

struct Molecule
{
    DenseGraph *graph;
    double *target;
    std::vector<std::pair<int, int>> edge;
    std::vector<std::string> label;
    int id;

    void build();
};