#include "Molecule.h"

#define numberOfFeatures 5

void Molecule::build()
{

    this->nFeatures = numberOfFeatures;

    for (int i = 0; i < edge.size(); ++i)
    {
        int u = edge[i].first;
        int v = edge[i].second;
        graph->adj[u][v] = 1;
        graph->adj[v][u] = 1;
    }

    for (int v = 0; v < graph->nVertices; ++v)
    {
        if (label[v] == "C")
        {
            graph->feature[v][0] = 1.0;
        }
        if (label[v] == "H")
        {
            graph->feature[v][1] = 1.0;
        }
        if (label[v] == "N")
        {
            graph->feature[v][2] = 1.0;
        }
        if (label[v] == "O")
        {
            graph->feature[v][3] = 1.0;
        }
        if (label[v] == "F")
        {
            graph->feature[v][4] = 1.0;
        }
    }
}
