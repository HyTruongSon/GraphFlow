#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <tuple>
#include "dirent.h"
#include "Molecule.cpp"
#include "../GraphFlow/DenseGraph.h"

/*
 * This corresponds to the number of
 * types of atoms in the molecules.
 */
#define numberOfFeatures 5

class MoleculeBuilder
{
public:
    MoleculeBuilder();
    Molecule **getMolecules();
    int getNumberOfMolecules();
    ~MoleculeBuilder(){};

private:
    Molecule **molecules;
    int numberOfMolecules;
    std::vector<std::pair<std::string, std::string>> readMoleculesFromDir();
    void initMoleculesArray();
    void buildMolecules(std::vector<std::pair<std::string, std::string>> &filePaths);
    Molecule *buildMolecule(std::vector<std::vector<double>> &adjecencyMatrix, std::vector<std::string> &labels, std::vector<double> &targets);
};
