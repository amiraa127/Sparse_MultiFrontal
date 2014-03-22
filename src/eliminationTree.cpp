#include "eliminationTree.hpp"

eliminationTree::eliminationTree(){
  numBlocks = 0;
  numCols = 0;
  numLevels = 0;
  root = NULL;
}

eliminationTree::eliminationTree(const int input_numCols,const int input_numBlocks,const std::vector<int> & rangVec,const std::vector<int> & treeVec){
    
  assert(input_numBlocks > 0);
  numBlocks = input_numBlocks;
  numCols = input_numCols;
  // Check inputs
  assert(treeVec[numBlocks - 1] == -1);
  assert(rangVec[numBlocks] == numCols);
  
  // Create elimination tree 
  numLevels = 1;
  createTree(rangVec,treeVec);
  analyzeTree(root);
  levelCols.resize(numLevels);
  for (int i = 0; i < numLevels; i++)
    levelCols[i] = countColsAtLevel(i,root);
  
  // Sanity check
  int sum = 0;
  for (int i = 0; i < numLevels; i++)
    sum += levelCols[i];
  assert(numCols == sum);
  int nodeLevelVecSize = nodeLevelVec.size();
  assert(nodeLevelVecSize == numLevels);
  
}

eliminationTree::~eliminationTree(){
  if (root != NULL)
    freeTree(root); 
}

void eliminationTree::freeTree(node* root){
  
  int numChildren = (root->children).size();
  if (numChildren == 0){
    delete(root);
    return;
  }
  for (int i = 0; i < numChildren; i++)
    freeTree((root->children)[i]);
  delete(root);
}

void eliminationTree::createTree(const std::vector<int> & rangVec,const std::vector<int> & treeVec){
  std::cout<<"creatingTree"<<std::endl;
  assert(treeVec.size() == numBlocks);
  assert(rangVec.size() == numBlocks + 1);
  std::vector<node*> blockPtrVec(numBlocks);
  std::vector<bool>  createdNode(numBlocks);
  // Initialize createdNode
  for (unsigned int i = 0; i < numBlocks; i++)
    createdNode[i] = false;

  for (unsigned int i = 0; i < numBlocks - 1; i++){
    //Child node
    node* currNodePtr;
    if (createdNode[i] == false){      
      currNodePtr= new node;
      blockPtrVec[i] = currNodePtr;
      createdNode[i] = true;
    }else{
      currNodePtr = blockPtrVec[i];
    }
    currNodePtr->min_Col = rangVec[i];
    currNodePtr->max_Col = rangVec[i + 1] - 1;
    currNodePtr->numCols = rangVec[i + 1] - rangVec[i];  
    currNodePtr->isLeaf = false;
    //Parent node
    int parentNodeIdx = treeVec[i];
    node* parentNodePtr;
    if (createdNode[parentNodeIdx] == false){
      parentNodePtr = new node;
      blockPtrVec[parentNodeIdx] = parentNodePtr;
      createdNode[parentNodeIdx] = true;
    }else{
      parentNodePtr = blockPtrVec[parentNodeIdx];
    }
    (parentNodePtr->children).push_back(currNodePtr);
   
  }
  // Complete the rootNode. It should have been created by now!
  root = blockPtrVec[numBlocks - 1];
  root->min_Col = rangVec[numBlocks - 1];
  root->max_Col = rangVec[numBlocks] - 1;
  root->numCols = rangVec[numBlocks] - rangVec[numBlocks - 1];
  root->currLevel = 0;
  root->isLeaf = false;
  
}

void eliminationTree::analyzeTree(node *root){
  
  int currLevel = root->currLevel;
  int nodeLevelVecSize = nodeLevelVec.size();
  if (nodeLevelVecSize < currLevel + 1){
    std::vector<node*> currLevelNodeVec;
    nodeLevelVec.push_back(currLevelNodeVec);
  }
  nodeLevelVec[currLevel].push_back(root);

  if (currLevel > (numLevels - 1))
    numLevels = currLevel + 1;

  int numChildren = (root->children).size(); 
  if (numChildren == 0){
    root->isLeaf = true;
    return;
  }
  
  for (int i = 0; i < numChildren; i++){
    node* child = (root->children)[i];
    child->currLevel = currLevel + 1;
    analyzeTree(child);
  }
}

int eliminationTree::countColsAtLevel(const int level,const node* root) const{
  if (root->currLevel == level)
    return root->numCols;
  if (root->isLeaf == true)
    return 0;
  int sum = 0;
  int numChildren = (root->children).size(); 
  for (int i = 0; i < numChildren; i++)
    sum += countColsAtLevel(level,(root->children)[i]);
  return sum;
}

