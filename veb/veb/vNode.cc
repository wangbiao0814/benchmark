#include "vNode.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

void InsertCSRNodeWithoutDumplicatedKey(CSRNode* csrNode, vEBKV* kv, int level, uint16_t tag)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    int csrNodeCount = GetCSRNodeCount(csrNode, keyBytes[level]);
    int csrNodeCumsum = GetCSRNodeCumsum(csrNode, keyBytes[level]);
    int blockIndex = (keyBytes[level] >> 6);
    int csrNodeTotalCount = GetCSRNodeTotalCount(csrNode, blockIndex);
    if(csrNodeTotalCount == 0)
    {
        csrNode->CSR[blockIndex] = (uint8_t*)calloc(1, 8 * (sizeof(vEBKV*)));
    }
    else if((csrNodeTotalCount & (0x7)) == 0)
    {
        uint8_t* newCSR = (uint8_t*)calloc(1, (csrNodeTotalCount + 8) * (sizeof(vEBKV*)));
        memcpy(newCSR, csrNode->CSR[blockIndex], csrNodeTotalCount * (sizeof(vEBKV*)));
        free(csrNode->CSR[blockIndex]);
        csrNode->CSR[blockIndex] = newCSR;
    }
    for(int i = csrNodeTotalCount - 1; i >= csrNodeCumsum + csrNodeCount; i--)
    {
        memcpy((uint8_t*)csrNode->CSR[blockIndex] + (i + 1) * (sizeof(vEBKV*)), (uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)), (sizeof(vEBKV*)));
    }
    *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (csrNodeCumsum + csrNodeCount) * (sizeof(vEBKV*))) = setLeafTag(kv, tag);
    AddCSRNodeCount(csrNode, keyBytes[level]);
}

void InsertBitmapLeafNodeWithoutDumplicatedKey(BitmapLeafNode* bitmapLeafNode, vEBKV* kv, int level)
{
    uint8_t key = kv->kv[level];
    int bitmapIndex = GetBitmapLeafNodeBitmapPopcnt(bitmapLeafNode, key);
    int bitmapTotal = GetBitmapLeafNodeBitmapPopcntAll(bitmapLeafNode);
    for(int i = bitmapTotal; i > bitmapIndex; i--) bitmapLeafNode->kvs[i] = bitmapLeafNode->kvs[i - 1];    
    SetBitmapLeafNodeBitmap(bitmapLeafNode, key);
    bitmapLeafNode->kvs[bitmapIndex] = kv;
}

void InsertCSRNodeOnly(CSRNode* csrNode, int blockIndex, int csrNodeCount, int csrNodeCumsum, vEBKV* kv, int level, uint16_t tag)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    int csrNodeTotalCount = GetCSRNodeTotalCount(csrNode, blockIndex);
    if(csrNodeTotalCount == 0)
    {
        csrNode->CSR[blockIndex] = (uint8_t*)calloc(1, 8 * (sizeof(vEBKV*)));
    }
    else if((csrNodeTotalCount & (0x7)) == 0)
    {
        uint8_t* newCSR = (uint8_t*)calloc(1, (csrNodeTotalCount + 8) * (sizeof(vEBKV*)));
        memcpy(newCSR, csrNode->CSR[blockIndex], csrNodeTotalCount * (sizeof(vEBKV*)));
        free(csrNode->CSR[blockIndex]);
        csrNode->CSR[blockIndex] = newCSR;
    }
    for(int i = csrNodeTotalCount - 1; i >= csrNodeCumsum + csrNodeCount; i--)
    {
        memcpy((uint8_t*)csrNode->CSR[blockIndex] + (i + 1) * (sizeof(vEBKV*)), (uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)), (sizeof(vEBKV*)));
    }
    *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (csrNodeCumsum + csrNodeCount) * (sizeof(vEBKV*))) = setLeafTag(kv, tag);
    AddCSRNodeCount(csrNode, keyBytes[level]);
}

int ConstructCSRNodeTree(vEB* root, CSRNode* csrNode, int blockIndex, int csrNodeCount, int csrNodeCumsum, vEBKV* kv, void* nodes[], int level, uint16_t tag)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    uint8_t lcp[MAX_KEY_LENGTH] = {0}; int lcpLength = kv->keyLength - level - 1;
    memcpy(lcp, keyBytes + level + 1, lcpLength);
    for(int i = csrNodeCumsum; i < csrNodeCumsum + csrNodeCount; i++)
    {
        vEBKV* leafTag = *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)));
        uint8_t* nodeKey = ((uint8_t*)((getLeaf(leafTag))->kv) + level + 1); int j;
        for(j = 0; j < lcpLength; j++)
        {
            if(nodeKey[j] != lcp[j]) break;
        }
        lcpLength = j;
    }
    //Find the longest common prefix, build upper and lower CSR node using the longest common prefix and a byte after the longest common prefix
    CSRNode* tempCSRNode = NULL; BitmapLeafNode* tempBitmapLeafNode = NULL; int levelIndex;
    for(levelIndex = level + 1; levelIndex <= level + lcpLength; levelIndex++)
    {
        if(isUpperLevel(levelIndex))
        {
            tempCSRNode = (CSRNode*)calloc(1, sizeof(CSRNode) + CSR_CHILD_INIT_SIZE * sizeof(void*));
        }
        else
        {
            tempCSRNode = (CSRNode*)calloc(1, sizeof(CSRNode));
        }
        tempCSRNode->type = CSR_NODE;
        SetCSRNodeBitmap(tempCSRNode, keyBytes[levelIndex]);
        nodes[levelIndex] = (void*)tempCSRNode;
    }
    if(levelIndex == root->maxKeyLength - 1)
    {
        tempBitmapLeafNode = (BitmapLeafNode*)calloc(1, sizeof(BitmapLeafNode) + 2 * sizeof(vEBKV*));
        tempBitmapLeafNode->type = BITMAP_LEAF_NODE;
        for(int i = csrNodeCumsum; i < csrNodeCumsum + csrNodeCount; i++)
        {
            vEBKV* leafTag = *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)));
            InsertBitmapLeafNodeWithoutDumplicatedKey(tempBitmapLeafNode, getLeaf(leafTag), levelIndex);
        }
        InsertBitmapLeafNodeWithoutDumplicatedKey(tempBitmapLeafNode, kv, levelIndex);
        nodes[levelIndex] = (void*)tempBitmapLeafNode;
    }
    else
    {
        if(isUpperLevel(levelIndex))
            tempCSRNode = (CSRNode*)calloc(1, sizeof(CSRNode) + CSR_CHILD_INIT_SIZE * sizeof(void*));
        else
            tempCSRNode = (CSRNode*)calloc(1, sizeof(CSRNode));  
        for(int i = csrNodeCumsum; i < csrNodeCumsum + csrNodeCount; i++)
        {
            vEBKV* leafTag = *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)));
            InsertCSRNodeWithoutDumplicatedKey(tempCSRNode, getLeaf(leafTag), levelIndex, getTag(leafTag));
        }
        InsertCSRNodeWithoutDumplicatedKey(tempCSRNode, kv, levelIndex, tag);
        nodes[levelIndex] = (void*)tempCSRNode;
    }
    return lcpLength;
}

void InsertCSRNodeTree(vEB* root, CSRNode* csrNode, void** parrentCSRNode, void* nodes[], vEBKV* kv, int level, int lcpLength)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    CSRNode* tempCSRNode = NULL;
    vEBHashKey hashKeyTemp; hashKeyTemp.end = kv->keyLength < MAX_KEY_LENGTH ? kv->keyLength : MAX_KEY_LENGTH;
    memcpy(hashKeyTemp.key, kv->kv, hashKeyTemp.end);
    for(int i = level + 1; i <= level + lcpLength + 1; i++)
    {
        if(isUpperLevel(i))
        {
            hashKeyTemp.end = i;
            (*((HashTableRoot*)root->levelHash[i >> 1]))[hashKeyTemp] = nodes[i];
            if(i == level + 1)
            {
                SetCSRNodeBitmap(csrNode, keyBytes[level]);
            }
        }
        else if(i == level + 1)
        {   
            int csrNodeBitmapPopcntAll = GetCSRNodeBitmapPopcntAll(csrNode);
            int csrNodeBitmapPopcnt = csrNodeBitmapPopcntAll > 128 ? keyBytes[level] : GetCSRNodeBitmapPopcnt(csrNode, keyBytes[level]);
            if(csrNodeBitmapPopcntAll >= 128)
            {
                if(csrNodeBitmapPopcntAll == 128)
                {
                    CSRNode* newCSRNode = (CSRNode*)calloc(1, sizeof(CSRNode) + 2 * csrNodeBitmapPopcntAll * sizeof(void*)); int idx = 0;
                    memcpy(newCSRNode, csrNode, sizeof(CSRNode));
                    for(int j = 0; j < 256; j++)
                    {
                        if(GetCSRNodeBitmap(csrNode, j))
                        {
                            newCSRNode->nodes[j] = csrNode->nodes[idx++];
                        }
                    }
                    free(csrNode);
                    csrNode = newCSRNode;
                    *parrentCSRNode = newCSRNode;
                }
                csrNode->nodes[keyBytes[level]] = nodes[i];
                SetCSRNodeBitmap(csrNode, keyBytes[level]);
            }
            else
            {
                if((csrNodeBitmapPopcntAll & (csrNodeBitmapPopcntAll - 1)) == 0 && csrNodeBitmapPopcntAll >= 4)
                {
                    CSRNode* newCSRNode = (CSRNode*)calloc(1, sizeof(CSRNode) + 2 * csrNodeBitmapPopcntAll * sizeof(void*)); 
                    memcpy(newCSRNode, csrNode, sizeof(CSRNode) + csrNodeBitmapPopcntAll * sizeof(void*));
                    free(csrNode);
                    csrNode = newCSRNode;
                    *parrentCSRNode = newCSRNode;
                }
                for(int j = csrNodeBitmapPopcntAll; j > csrNodeBitmapPopcnt; j--) csrNode->nodes[j] = csrNode->nodes[j - 1];
                csrNode->nodes[csrNodeBitmapPopcnt] = nodes[i];
                SetCSRNodeBitmap(csrNode, keyBytes[level]);
            }
        }
        else
        {
            tempCSRNode = (CSRNode*)nodes[i - 1]; tempCSRNode->nodes[0] = nodes[i];
        }
    }
}

void DeleteCSRNodeKeys(CSRNode* csrNode, int blockIndex, int csrNodeCount, int csrNodeCumsum, vEBKV* kv, int level)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    int csrNodeTotalCount = GetCSRNodeTotalCount(csrNode, blockIndex);
    int csrNodeCapacity = (((csrNodeTotalCount - 1) >> 3) << 3) + 8;

    for(int i = csrNodeCumsum + csrNodeCount; i < csrNodeTotalCount; i++)
    {
        memcpy((uint8_t*)csrNode->CSR[blockIndex] + (i - csrNodeCount) * (sizeof(vEBKV*)) , (uint8_t*)csrNode->CSR[blockIndex] + i * (sizeof(vEBKV*)), sizeof(vEBKV*));
    }
    AddCSRNodeCount(csrNode, keyBytes[level]); csrNode->csrRank[keyBytes[level] >> 6] -= 2; csrNodeTotalCount -= 1;
    int csrNodeNewCapacity = (((csrNodeTotalCount - 1) >> 3) << 3) + 8;
    if(csrNodeTotalCount == 0)
    {
        free(csrNode->CSR[blockIndex]); csrNode->CSR[blockIndex] = NULL;
    }
    else if(csrNodeNewCapacity < csrNodeCapacity)
    {
        uint8_t* newCSR = (uint8_t*)calloc(1, (csrNodeTotalCount) * (sizeof(vEBKV*)));
        memcpy(newCSR, csrNode->CSR[blockIndex], csrNodeTotalCount * (sizeof(vEBKV*)));
        free(csrNode->CSR[blockIndex]);
        csrNode->CSR[blockIndex] = newCSR;
    }
}

void InsertCSRNode(vEB* root, CSRNode* csrNode, void** parrentCSRNode, vEBKV* kv, int level, uint16_t tag)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;

    int csrNodeCount = GetCSRNodeCount(csrNode, keyBytes[level]);
    int csrNodeCumsum = GetCSRNodeCumsum(csrNode, keyBytes[level]);
    int blockIndex = (keyBytes[level] >> 6);

    for(int i = csrNodeCumsum; i < csrNodeCumsum + csrNodeCount; i++)
    {
        vEBKV* leafTag = *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)));
        if(getTag(leafTag) == tag && !memcmp(kv->kv, (vEBKV*)(getLeaf(leafTag))->kv, kv->keyLength))
        {
            *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*))) = kv;
            return;
        }
    }

    if(csrNodeCount == 1)
    {
        void* nodes[MAX_KEY_LENGTH] = {NULL};
        int lcpLength = ConstructCSRNodeTree(root, csrNode, blockIndex, csrNodeCount, csrNodeCumsum, kv, nodes, level, tag);
        InsertCSRNodeTree(root, csrNode, parrentCSRNode, nodes, kv, level, lcpLength);
        DeleteCSRNodeKeys((CSRNode*)*parrentCSRNode, blockIndex, csrNodeCount, csrNodeCumsum, kv, level);
        return;
    }
    
    if(csrNodeCount == 0 && GetCSRNodeBitmap(csrNode, keyBytes[level]))
    {
        int csrNodeBitmapPopcntAll = GetCSRNodeBitmapPopcntAll(csrNode);
        int csrNodeBitmapPopcnt = csrNodeBitmapPopcntAll > 128 ? keyBytes[level] : GetCSRNodeBitmapPopcnt(csrNode, keyBytes[level]) - 1;
        
        BaseNode* baseNode = (BaseNode*)csrNode->nodes[csrNodeBitmapPopcnt];
        switch (baseNode->type)
        {
        case CSR_NODE:
            InsertCSRNode(root, (CSRNode*)baseNode, &csrNode->nodes[csrNodeBitmapPopcnt], kv, level + 1, tag);
            break;
        case BITMAP_LEAF_NODE:
            InsertBitmapLeafNode((BitmapLeafNode*)baseNode, &csrNode->nodes[csrNodeBitmapPopcnt], kv, level + 1);
            break;
        }
        return;
    }   
    InsertCSRNodeOnly(csrNode, blockIndex, csrNodeCount, csrNodeCumsum, kv, level, tag);
}

void* GetCSRNode(CSRNode* csrNode, vEBKV* kv, int level, uint16_t tag)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;

    int csrNodeCount = GetCSRNodeCount(csrNode, keyBytes[level]);
    int csrNodeCumsum = GetCSRNodeCumsum(csrNode, keyBytes[level]);
    int blockIndex = (keyBytes[level] >> 6);
    
    if(csrNodeCount == 0 && GetCSRNodeBitmap(csrNode, keyBytes[level]))
    {
        int csrNodeBitmapPopcntAll = GetCSRNodeBitmapPopcntAll(csrNode);
        int csrNodeBitmapPopcnt = csrNodeBitmapPopcntAll > 128 ? keyBytes[level] : GetCSRNodeBitmapPopcnt(csrNode, keyBytes[level]) - 1;
        
        BaseNode* baseNode = (BaseNode*)csrNode->nodes[csrNodeBitmapPopcnt];
        switch (baseNode->type)
        {
        case CSR_NODE:
            return GetCSRNode((CSRNode*)baseNode, kv, level + 1, tag);
        case BITMAP_LEAF_NODE:
            return GetBitmapLeafNode((BitmapLeafNode*)baseNode, kv, level + 1);
        }
    }

    for(int i = csrNodeCumsum; i < csrNodeCumsum + csrNodeCount; i++)
    {
        vEBKV* leafTag = *(vEBKV**)((uint8_t*)csrNode->CSR[blockIndex] + (i) * (sizeof(vEBKV*)));
        if(getTag(leafTag) == tag && !memcmp(kv->kv, (vEBKV*)(getLeaf(leafTag))->kv, kv->keyLength))
        {
            return getLeaf(leafTag);
        }
    }
    return NULL;
}

void CollectCSRNode(CSRNode* csrNode, int level, vEBStatistics* vs)
{
    int keyCount = 0;
    for(int i = 0; i < BLOCKS; i++) keyCount += GetCSRNodeTotalCount(csrNode, i);
    vs->vNodeKeysDistribution[level] += keyCount;
    if(isUpperLevel(level))
    {
        for(int i = 0; i < 256; i++)
        {    
            if(GetCSRNodeCount(csrNode, i) == 0 && GetCSRNodeBitmap(csrNode, i))
            {
                int csrNodeBitmapPopcntAll = GetCSRNodeBitmapPopcntAll(csrNode);
                int csrNodeBitmapPopcnt = csrNodeBitmapPopcntAll > 128 ? i : GetCSRNodeBitmapPopcnt(csrNode, i) - 1;
                
                BaseNode* baseNode = (BaseNode*)csrNode->nodes[csrNodeBitmapPopcnt];
                switch (baseNode->type)
                {
                case CSR_NODE:
                    CollectCSRNode((CSRNode*)baseNode, level + 1, vs);
                    break;
                case BITMAP_LEAF_NODE:
                    CollectBitmapLeafNode((BitmapLeafNode*)baseNode, level + 1, vs);
                    break;
                }
            }

        }
    }    
}

void InsertBitmapLeafNode(BitmapLeafNode* bitmapLeafNode, void** parentBitmapLeafNode, vEBKV* kv, int level)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    int bitmapLeafNodePopcntAll = GetBitmapLeafNodeBitmapPopcntAll(bitmapLeafNode);
    int bitmapLeafNodePopcnt = bitmapLeafNodePopcntAll > 128 ? keyBytes[level] : GetBitmapLeafNodeBitmapPopcnt(bitmapLeafNode, keyBytes[level]);

    if(GetBitmapLeafNodeBitmap(bitmapLeafNode, keyBytes[level]))
    {
        bitmapLeafNode->kvs[bitmapLeafNodePopcnt - 1] = kv;
    }
    else
    {   
        if(bitmapLeafNodePopcntAll >= 128)
        {
            if(bitmapLeafNodePopcntAll == 128)
            {
                BitmapLeafNode* newBitmapLeafNode = (BitmapLeafNode*)calloc(1, sizeof(BitmapLeafNode) + 2 * bitmapLeafNodePopcntAll * sizeof(vEBKV*)); int idx = 0;
                memcpy(newBitmapLeafNode, bitmapLeafNode, sizeof(BitmapLeafNode));
                for(int i = 0; i < 256; i++)
                {
                    if(GetBitmapLeafNodeBitmap(bitmapLeafNode, i))
                    {
                        newBitmapLeafNode->kvs[i] = bitmapLeafNode->kvs[idx++];
                    }
                }
                free(bitmapLeafNode);
                bitmapLeafNode = newBitmapLeafNode;
                *parentBitmapLeafNode = newBitmapLeafNode;
            }
            bitmapLeafNode->kvs[keyBytes[level]] = kv;
            SetBitmapLeafNodeBitmap(bitmapLeafNode, keyBytes[level]);
        }
        else
        {
            if((bitmapLeafNodePopcntAll & (bitmapLeafNodePopcntAll - 1)) == 0)
            {
                BitmapLeafNode* newBitmapLeafNode = (BitmapLeafNode*)calloc(1, sizeof(BitmapLeafNode) + 2 * bitmapLeafNodePopcntAll * sizeof(vEBKV*));
                memcpy(newBitmapLeafNode, bitmapLeafNode, sizeof(BitmapLeafNode) + bitmapLeafNodePopcntAll * sizeof(vEBKV*));
                free(bitmapLeafNode);
                bitmapLeafNode = newBitmapLeafNode;
                *parentBitmapLeafNode = newBitmapLeafNode;
            }
            for(int i = bitmapLeafNodePopcntAll; i > bitmapLeafNodePopcnt; i--) bitmapLeafNode->kvs[i] = bitmapLeafNode->kvs[i - 1];
            bitmapLeafNode->kvs[bitmapLeafNodePopcnt] = kv;
            SetBitmapLeafNodeBitmap(bitmapLeafNode, keyBytes[level]);
        }

    }
}

void* GetBitmapLeafNode(BitmapLeafNode* bitmapLeafNode, vEBKV* kv, int level)
{
    uint8_t* keyBytes = (uint8_t*)kv->kv;
    if(GetBitmapLeafNodeBitmap(bitmapLeafNode, keyBytes[level]))
    {
        int bitmapLeafNodePopcntAll = GetBitmapLeafNodeBitmapPopcntAll(bitmapLeafNode);
        int bitmapLeafNodePopcnt = bitmapLeafNodePopcntAll > 128 ? keyBytes[level] : GetBitmapLeafNodeBitmapPopcnt(bitmapLeafNode, keyBytes[level]) - 1;
        return bitmapLeafNode->kvs[bitmapLeafNodePopcnt];
    }
    return 0;
}

void CollectBitmapLeafNode(BitmapLeafNode* bitmapLeafNode, int level, vEBStatistics* vs)
{
    vs->vNodeKeysDistribution[level] += GetBitmapLeafNodeBitmapPopcntAll(bitmapLeafNode);
}