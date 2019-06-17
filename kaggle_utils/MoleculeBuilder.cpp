#include "MoleculeBuilder.h"

MoleculeBuilder::MoleculeBuilder()
{
    std::vector<std::string> filePaths = this->readMoleculesFromDir();
    this->numberOfMolecules = filePaths.size();
    std::cout << "There are " << this->numberOfMolecules << " molecules overall." << std::endl;
    this->initMoleculesArray();
    this->buildMolecules(filePaths);
};

Molecule **MoleculeBuilder::getMolecules()
{
    return this->molecules;
}

int MoleculeBuilder::getNumberOfMolecules()
{
    return this->numberOfMolecules;
}

std::vector<std::string> MoleculeBuilder::readMoleculesFromDir()
{
    std::vector<std::string> filePaths;
    std::string dirPath = "../kaggle_utils/molecules/train/";
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dirPath.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string fileName = ent->d_name;
            if (fileName.find("adj_mat") != std::string::npos)
            {
                filePaths.push_back(dirPath + fileName);
            }
        }
    }
    else
    {
        perror("Could not open directory.");
        exit(1);
    }
    closedir(dir);

    return filePaths;
}

void MoleculeBuilder::initMoleculesArray()
{
    this->molecules = new Molecule *[this->numberOfMolecules];
    for (int i = 0; i < this->numberOfMolecules; ++i)
    {
        molecules[i] = new Molecule();
    }
}

void MoleculeBuilder::buildMolecules(std::vector<std::string> &filePaths)
{
    for (int counter = 0; counter < this->numberOfMolecules; ++counter)
    {
        std::ifstream infile(filePaths[counter], std::ios::binary);
        if (!infile.is_open())
        {
            std::cout << "Failed to open " << filePaths[counter] << "\n";
        }
        else
        {
            /*
                * First line contains the structure in form like : # C C N H H H
                * thus I need not count the hashtag and collect the atoms.               
                */
            std::vector<std::string> labels;
            std::string line;
            std::getline(infile, line);
            std::istringstream iss(line);
            std::string value;
            while (iss >> value)
            {
                if (value != "#")
                {
                    labels.push_back(value);
                }
            }
            /*
                 * Then proceed with the adjecency matrix.
                 */
            std::vector<std::vector<double>> adjecencyMatrix;
            while (!infile.eof())
            {
                std::string line;
                std::getline(infile, line);
                std::istringstream iss(line);
                std::vector<double> adjecencyVec;
                double value;
                while (iss >> value)
                {
                    adjecencyVec.push_back(value);
                }
                adjecencyMatrix.push_back(adjecencyVec);
            }
            molecules[counter] = this->buildMolecule(adjecencyMatrix, labels);
        }
    }
}

Molecule *MoleculeBuilder::buildMolecule(std::vector<std::vector<double>> &adjecencyMatrix, std::vector<std::string> &labels)
{
    Molecule *molecule = new Molecule();
    molecule->graph = new DenseGraph(labels.size(), numberOfFeatures);
    // TODO: update it with appropriate target
    molecule->target = new double[3];
    molecule->target[0] = 1.;
    molecule->target[1] = 2.;
    molecule->target[2] = 3.;

    molecule->edge.clear();
    molecule->label.clear();
    for (int i = 0; i < labels.size(); ++i)
    {
        /*
         * Create the graph by pushing edges to it
         * according to the adjecency matrix of the
         * molecular structure.
        */
        for (int j = i + 1; j < labels.size() - 1; ++j)
        {
            if (adjecencyMatrix[i][j] > 0.)
            {
                molecule->edge.push_back(std::make_pair(i, j));
            }
        }
        /*
         * Also adding labels.
        */
        molecule->label.push_back(labels[i]);
    }
    molecule->build();
    return molecule;
};