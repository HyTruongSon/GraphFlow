#include "MoleculeBuilder.h"

MoleculeBuilder::MoleculeBuilder()
{
    std::vector<std::pair<std::string, std::string>> adjPaths = this->readMoleculesFromDir();
    this->numberOfMolecules = adjPaths.size();
    std::cout << "There are " << this->numberOfMolecules << " molecules overall." << std::endl;
    this->initMoleculesArray();
    this->buildMolecules(adjPaths);
};

Molecule **MoleculeBuilder::getMolecules()
{
    return this->molecules;
}

int MoleculeBuilder::getNumberOfMolecules()
{
    return this->numberOfMolecules;
}

std::vector<std::pair<std::string, std::string>> MoleculeBuilder::readMoleculesFromDir()
{
    std::vector<std::string> adjPaths;
    std::vector<std::string> labelPaths;

    std::string bondsDirPath = "../kaggle_utils/molecules/bonds/";
    std::string labelsDirPath = "../kaggle_utils/molecules/labels/";

    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(bondsDirPath.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string fileName = ent->d_name;
            if (fileName.find("adj_mat") != std::string::npos)
            {
                adjPaths.push_back(bondsDirPath + fileName);
            }
        }
    }
    else
    {
        perror("Could not open bonds directory.");
        exit(1);
    }
    closedir(dir);

    if ((dir = opendir(labelsDirPath.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string fileName = ent->d_name;
            if (fileName.find("dsgdb9nsd") != std::string::npos)
            {
                labelPaths.push_back(labelsDirPath + fileName);
            }
        }
    }
    else
    {
        perror("Could not open labels directory.");
        exit(1);
    }
    closedir(dir);

    std::vector<std::pair<std::string, std::string>> filePaths;
    for (int i = 0; i < labelPaths.size(); ++i)
    {
        filePaths.push_back(std::make_pair(adjPaths[i], labelPaths[i]));
    }

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

void MoleculeBuilder::buildMolecules(std::vector<std::pair<std::string, std::string>> &filePaths)
{
    for (int counter = 0; counter < this->numberOfMolecules; ++counter)
    {
        std::ifstream adj_infile(filePaths[counter].first, std::ios::binary);
        std::ifstream tar_infile(filePaths[counter].second, std::ios::binary);
        if (!adj_infile.is_open() && !tar_infile.is_open())
        {
            std::cout << "Failed to open files : " << filePaths[counter].first;
            std::cout << " " << filePaths[counter].second << "\n";
        }
        else
        {
            /*
                * First line contains the structure in form like : # C C N H H H
                * thus I need not count the hashtag and collect the atoms.               
                */
            std::vector<std::string> labels;
            std::string line;
            std::getline(adj_infile, line);
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
            while (!adj_infile.eof())
            {
                std::string line;
                std::getline(adj_infile, line);
                std::istringstream iss(line);
                std::vector<double> adjecencyVec;
                double value;
                while (iss >> value)
                {
                    adjecencyVec.push_back(value);
                }
                adjecencyMatrix.push_back(adjecencyVec);
            }

            std::vector<double> targets;
            while (!tar_infile.eof())
            {
                std::string target;
                std::getline(tar_infile, target);
                std::istringstream iss(target);
                double value;
                while (iss >> value)
                {
                    targets.push_back(value);
                }
            }

            molecules[counter] = this->buildMolecule(adjecencyMatrix, labels, targets);
        }
    }
}

Molecule *MoleculeBuilder::buildMolecule(std::vector<std::vector<double>> &adjecencyMatrix, std::vector<std::string> &labels, std::vector<double> &targets)
{
    Molecule *molecule = new Molecule();
    molecule->graph = new DenseGraph(labels.size(), numberOfFeatures);
    // TODO: update it with appropriate target
    molecule->target = new double[targets.size()];
    for (int i = 0; i < targets.size(); ++i)
    {
        molecule->target[i] = targets[i];
    }

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