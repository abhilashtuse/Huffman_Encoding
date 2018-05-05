#include <iostream>
#include <queue>
#include "binary_tree.h"

using namespace std;
/* Compute the "height" of a tree -- the number of
    nodes along the longest path from the root node
    down to the farthest leaf node.*/
int height(Node* node)
{
    if (node==NULL)
        return 0;
    else
    {
        /* compute the height of each subtree */
        int lheight = height(node->left);
        int rheight = height(node->right);

        /* use the larger one */
        if (lheight > rheight)
            return(lheight+1);
        else return(rheight+1);
    }
}


// Function to allocate a new tree node
Node* getNode(char ch, int freq, Node* left, Node* right)
{
    Node* node = new Node();

    node->ch = ch;
    node->freq = freq;
    node->left = left;
    node->right = right;

    return node;
}


void convertTreeToArray(Node *root, char *arr, int treeHeight)
{
    // Base Case
    if (root == NULL)  return;

    // Create an empty queue for level order tarversal
    queue<Node *> q;

    Node *marker = getNode('$', 0, nullptr, nullptr);
    // Enqueue Root and initialize height
    q.push(root);

    int counter = 0, ind = 0;
    while (1)
    {
        // nodeCount (queue size) indicates number of nodes
        // at current lelvel.
        int nodeCount = q.size();
        if (nodeCount == 0)
            break;

        // Dequeue all nodes of current level and Enqueue all
        // nodes of next level
        while (nodeCount > 0)
        {
            Node *node = q.front();
            if (node->ch == '\0') {
               printf("* "); arr[ind] = '*'; ind++;
            }
            else {
               printf("%c ", node->ch); arr[ind] = node->ch; ind++;
            }
            q.pop();
            if (node->left != NULL)
                q.push(node->left);
            else if (counter < treeHeight)
                q.push(marker);
            if (node->right != NULL)
                q.push(node->right);
            else if (counter < treeHeight)
                q.push(marker);

            nodeCount--;
        }
        cout << endl;
        counter++;
    }
}
