#include "vEB.h"
#include "vNode.h"



void SearchLongestPrefixInHashTable(vEB* root, vEBKV* kv, int& idx, HashTableRoot::iterator& iter, vEBHashKey& hashKey)
{
    /*
        We need to find the maximum i (0<i<(kv->keyLength+1) / 2 ) that satisfies the first 2i bytes of the key are in root->levelHash[i]
        [1, 4) mid = 2 - > [3, 4) mid = 3 - > [4, 4) idx = 3
                                          - > [3, 3) idx = 2
                       - > [1, 2) mid = 1 - > [2, 2) idx = 1
                                          - > [1, 1] idx = -1
    */ 
    HashTableRoot::iterator nodePointerIterTemp;
    int start = 1, end = (kv->keyLength + 1) / 2; // i in [start, end)
    while(start < end)
    {
        int mid = start + (end - start) / 2; hashKey.end = mid * 2; //check if first 2*mid bytes of key are in the hashtable
        if(!((HashTableRoot*)root->levelHash[mid])->empty() && (nodePointerIterTemp = ((HashTableRoot*)root->levelHash[mid])->find(hashKey)) != ((HashTableRoot*)root->levelHash[mid])->end())
        {
            start = mid + 1; idx = mid; iter = nodePointerIterTemp;
        }
        else
        {
            end = mid;
        }
    }
}

void SearchLongestPrefixInHashTableForRange(vEB* root, vEBIterator* iter, vEBKV* kv, int& idx, vEBHashKey& hashKey)
{
    HashTableRoot::iterator nodePointerIterTemp;
    int start = 1, end = (kv->keyLength + 1) / 2; // i in [start, end)
    while(start < end)
    {
        int mid = start + (end - start) / 2; hashKey.end = mid * 2; //check if first 2*mid bytes of key are in the hashtable
        if(!((HashTableRoot*)root->levelHash[mid])->empty() && (nodePointerIterTemp = ((HashTableRoot*)root->levelHash[mid])->find(hashKey)) != ((HashTableRoot*)root->levelHash[mid])->end())
        {
            start = mid + 1; idx = mid; iter->currentLevel = mid * 2; iter->levelvNodes[iter->currentLevel] = nodePointerIterTemp->second;
        }
        else
        {
            end = mid;
        }
    }
}


void InitvEB(vEB* root, int maxKeyLength) //finish
{
    if(maxKeyLength > MAX_KEY_LENGTH)
    {
        assert(0 && "max key length cannot exceed 16!");
    }
    root->levelHash = (void**)calloc(1, maxKeyLength * sizeof(void*));
    /*
    // We use the direct pointers to index first two bytes of keys, and the hashtables to index the first four, six ... bytes of keys
    for(int i = 1; i < (maxKeyLength + 1) / 2; i++)
    {
        if(i == 1) 
            root->levelHash[i] = (void*)calloc(1, sizeof(vEBDirectPointer));
        else
            root->levelHash[i] = (void*)(new HashTableRoot(256));
    }
    */
    // We use hashtable to index first two, four, six ... bytes of keys
    for(int i = 1; i < (maxKeyLength + 1) / 2; i++)
    {
        root->levelHash[i] = (void*)(new HashTableRoot(256));
    }
    root->maxKeyLength = maxKeyLength;
}


void InsertvEB(vEB* root, vEBKV* kv) 
{
    vEBHashKey hashKey; hashKey.end = kv->keyLength < MAX_KEY_LENGTH ? kv->keyLength : MAX_KEY_LENGTH;
    memcpy(hashKey.key, kv->kv, hashKey.end);

    int idx = -1; HashTableRoot::iterator iter;
    SearchLongestPrefixInHashTable(root, kv, idx, iter, hashKey);
    hashKey.end = kv->keyLength; uint16_t tag = (hash_value(hashKey) & 0xFFFF);
    if(idx == -1)
    {
        CSRNode* csrNode = (CSRNode*)calloc(1, sizeof(CSRNode) + CSR_CHILD_INIT_SIZE * sizeof(void*));
        InsertCSRNodeWithoutDumplicatedKey(csrNode, kv, 2, tag);
        hashKey.end = 2; (*((HashTableRoot*)root->levelHash[1]))[hashKey] = (void*)csrNode;
    }
    else
    {
        InsertCSRNode(root, (CSRNode*)(iter->second), &(iter->second), kv, idx * 2, tag);
    }
    
}

void* GetvEB(vEB* root, vEBKV* kv) 
{
    vEBHashKey hashKey; hashKey.end = kv->keyLength < MAX_KEY_LENGTH ? kv->keyLength : MAX_KEY_LENGTH;
    memcpy(hashKey.key, kv->kv, hashKey.end);
    
    int idx = -1; HashTableRoot::iterator iter;
    SearchLongestPrefixInHashTable(root, kv, idx, iter, hashKey);
    if(idx == -1) return NULL;
    hashKey.end = kv->keyLength; uint16_t tag = (hash_value(hashKey) & 0xFFFF);
    return GetCSRNode((CSRNode*)(iter->second), kv, idx * 2, tag);
}


void CollectvEB(vEB* root, vEBStatistics* vs)
{
    for(int i = 1; i < (root->maxKeyLength + 1) / 2; i++)
    {
        HashTableRoot* tempHash = (HashTableRoot*)root->levelHash[i];
        HashTableRoot::iterator tempHashIter;
        for(tempHashIter = tempHash->begin(); tempHashIter != tempHash->end(); tempHashIter++)
        {
            CollectCSRNode((CSRNode*)(tempHashIter->second), i * 2, vs);
        }
    }
}
