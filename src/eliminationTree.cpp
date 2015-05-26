#include "eliminationTree.hpp"

eliminationTree::eliminationTree(){
  numBlocks = 0;
  numCols = 0;
  numLevels = 0;
  root = NULL;
}

eliminationTree::eliminationTree(const int* colPtr,const int* row_ind,const int input_NumCols){
  std::vector<int> parentVec = createParentVec(colPtr,row_ind,input_NumCols);
  std::vector<int> rangVec   = createRangVec(parentVec,input_NumCols);
  std::vector<int> treeVec   = createTreeVec(parentVec,rangVec);
  std::cout<<treeVec.size()<<std::endl;
  build_eliminationTree(input_NumCols,treeVec.size(),rangVec,treeVec);
}


eliminationTree::eliminationTree(const int input_numCols,const int input_numBlocks,const std::vector<int> & rangVec,const std::vector<int> & treeVec){
  build_eliminationTree(input_numCols,input_numBlocks,rangVec,treeVec);
}

eliminationTree::~eliminationTree(){
  if (root != NULL)
    freeTree(root); 
}


void eliminationTree::build_eliminationTree(const int input_numCols,const int input_numBlocks,const std::vector<int> & rangVec,const std::vector<int> & treeVec){
  assert(input_numBlocks > 0);
  numBlocks = input_numBlocks;
  numCols = input_numCols;
  // Check inputs
  assert(treeVec[numBlocks - 1] == -1);
  assert(rangVec[numBlocks] == numCols);
  
  // Create elimination tree 
  numLevels = 1;
  if (numBlocks > 1){
    createTree(rangVec,treeVec);
    analyzeTree(root);
    levelCols.resize(numLevels);
    for (int i = 0; i < numLevels; i++)
      levelCols[i] = countColsAtLevel(i,root);
  }else{
    root = new node;
    root->isLeaf = true;
    root->currLevel = 0;
    root->numCols = numCols;
    root->min_Col = 0;
    root->max_Col = numCols - 1;
    levelCols.push_back(numCols);
    std::vector<node*> rootPtr;
    rootPtr.push_back(root);
    nodeLevelVec.push_back(rootPtr);
  }
  // Sanity check
#ifndef NDEBUG
  int sum = 0;
  for (int i = 0; i < numLevels; i++)
    sum += levelCols[i];
  assert(numCols == sum);
  int nodeLevelVecSize = nodeLevelVec.size();
  assert(nodeLevelVecSize == numLevels);
#endif
}


std::vector<int> eliminationTree::createParentVec(const int* colPtr,const int* row_ind,const int input_NumCols){

  int n = input_NumCols;
  std::vector<int> parent(n,0);
  std::vector<int> ancestor(n,0);
  
  for (int k = 0; k < n; k++){
    parent[k] = -1;
    ancestor[k] = -1;
    for (int p = colPtr[k]; p < colPtr[k+1]; p++){
      int i = row_ind[p];
      while ((i != -1) && (i < k)){
	int inext   = ancestor[i]; // Search for the root
	ancestor[i] = k;           // Update ancestor for efficiency
	if (inext == -1)           
	  parent[i] = k;           // Connect to k
	i = inext;
      }
    } 
  }

  // Correct for multiple roots
  std::vector<int> rootIdx;
  for (int i = 0; i < input_NumCols; i++)
    if (parent[i] < 0)
      rootIdx.push_back(i);
  
  for (size_t i = 0; i < (rootIdx.size()- 1); i++){
    parent[rootIdx[i]] = rootIdx[rootIdx.size() - 1];
  }
  //
  
  return parent;
}

std::vector<int> eliminationTree::createRangVec(const std::vector<int> & parentVec,const int input_NumCols){

 
  std::vector<int> childrenVec(input_NumCols,0);
  std::vector<int> rangVec;
  int rootNumChildren = 0;

  for (int i = 0; i < input_NumCols; i++){
    if (parentVec[i] >= 0)
      childrenVec[parentVec[i]] ++;
    else
      rootNumChildren++;
  }
  
  rangVec.push_back(0);
  int numBlocks = 0;
  int currBlockStartIdx = 0;
  int currBlockEndIdx   = 0;
  while (currBlockEndIdx < (input_NumCols - 1)){
    bool cond1 = (parentVec[currBlockEndIdx + 1] == parentVec[currBlockEndIdx]);
    bool cond2 = (parentVec[currBlockEndIdx] == currBlockEndIdx + 1) && (childrenVec[currBlockEndIdx + 1] == 1);
    bool cond3 = (parentVec[currBlockEndIdx] == -1) && (rootNumChildren == 1);
    if (cond1 || cond2 || cond3)
      currBlockEndIdx ++;
    else{
      rangVec.push_back(currBlockEndIdx + 1);
      currBlockStartIdx = currBlockEndIdx + 1;
      currBlockEndIdx   = currBlockStartIdx;
      numBlocks ++;
    }
  }
  if (rangVec[rangVec.size() - 1] != input_NumCols){
    rangVec.push_back(input_NumCols);
    numBlocks ++;
  }
  return rangVec;
}

std::vector<int> eliminationTree::createTreeVec(const std::vector<int> & parentVec,std::vector<int> & rangVec){
  numBlocks = rangVec.size() - 1;
  std::vector<int> treeVec(numBlocks);
  std::vector<int>::iterator up;

  for (unsigned int i = 0;i < numBlocks; i++){
    int endBlkIdx = rangVec[i + 1] - 1;
    int indParent = parentVec[endBlkIdx];
    if (indParent == -1){
      treeVec[i] = -1;
    }
    else{
      up = std::upper_bound(rangVec.begin(),rangVec.end(),indParent);
      treeVec[i] = (up - rangVec.begin()) - 1;
    }
  }
  return treeVec;
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

void eliminationTree::test(const int* colPtr,const int* row_ind,const int input_NumCols){
  std::vector<int> parentVec = createParentVec(colPtr,row_ind,input_NumCols);
  std::vector<int> rangVec   = createRangVec(parentVec,input_NumCols);
  std::vector<int> treeVec   = createTreeVec(parentVec,rangVec);
  build_eliminationTree(input_NumCols,treeVec.size(),rangVec,treeVec);
  for (unsigned int i = 0; i < treeVec.size(); i++)
    std::cout<<rangVec[i]<<" "<<treeVec[i]<<std::endl;
  std::cout<<rangVec[treeVec.size()]<<std::endl;
  std::cout<<"done"<<std::endl;

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

