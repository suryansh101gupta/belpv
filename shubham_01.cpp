#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int V;
vector<vector<int>> adj;

// bool edgeExists(int u, int v)
// {
//     for (int nbr : adj[u])
//     {
//         if (nbr == v)
//             return true;
//     }
//     return false;
// }

// void addEdge(int u, int v)
// {
//     if (!edgeExists(u, v))
//     {
//         adj[u].push_back(v);
//         adj[v].push_back(u);
//     }
// }

void addEdge(int u, int v){
    adj[u].push_back(v);
    adj[v].push_back(u);
}

void dfsUtil(int node, vector<char>& visited){
    visited[node] = 1;

    for (auto nbr : adj[node]){
        if (!visited[nbr]){
            dfsUtil(nbr, visited);
        }
    }
}

void dfs(int start){
    vector<char> visited(V, 0);
    dfsUtil(start, visited);
}

// void dfsUtil(int node, vector<bool>& visited){
//     visited[node] = 1;

//     for (int i = 0; i < V; i++){
//         if (!visited[i]){
//             dfsUtil(i, visited);
//         }
//     }
// }

// void dfs(int start){
//     vector<bool> visited(V, false);
//     dfsUtil(start, visited);
// }

void dfsUtilParallel(int node, vector<char>& visited){
    for (auto nbr : adj[node]){
        if (!visited[nbr]){
            bool spawn_task = false;

            #pragma omp critical
            {
                if (!visited[nbr]){
                    visited[nbr] = 1;
                    spawn_task = true;
                }
            }

            if (spawn_task){
                #pragma omp task shared(visited)
                dfsUtilParallel(nbr, visited);
            }
        }
    }

    #pragma omp taskwait
}

void dfsParallel(int start){
    vector<char> visited(V, 0);
    visited[start] = 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            dfsUtilParallel(start, visited);
        }
    }
}

void bfs(int start){
    vector<char> visited(V, 0);
    queue<int> q;
    q.push(start);
    visited[start] = 1;

    while (!q.empty()){
        int front = q.front();
        q.pop();
        for (auto nbr : adj[front]){
            if (!visited[nbr]){
                visited[nbr] = 1;
                q.push(nbr);
            }
        }
    }
}

void bfsParallel(int start){
    vector<char> visited(V, 0);
    vector<int> current;
    current.push_back(start);
    visited[start] = 1;

    while (!current.empty()){
        vector<int> next;

        #pragma omp parallel for
        for (int i = 0; i < current.size(); i++){
            int node = current[i];
            for (auto nbr : adj[node]){
                if (!visited[nbr]){
                    
                    #pragma omp critical
                    {
                        if (!visited[nbr]){
                            visited[nbr] = 1;
                            next.push_back(nbr);
                        }
                    }
                }
            }
        }

        current = next;
    }
}

int main(){
    V = 20000;

    adj.resize(V);

    srand(42);

    // Generate random graph
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < 150; j++)
        {
            int nbr = rand() % V;
            if (nbr != i)
            {
                addEdge(i, nbr);
            }
        }
    }

    double t1, t2;

    // Parallel DFS
    t1 = omp_get_wtime();
    dfsParallel(0);
    t2 = omp_get_wtime();
    cout << "Time taken by Parallel DFS: "
         << t2 - t1 << " seconds" << endl;

    // Parallel BFS
    t1 = omp_get_wtime();
    bfsParallel(0);
    t2 = omp_get_wtime();
    cout << "Time taken by Parallel BFS: "
         << t2 - t1 << " seconds" << endl;

    // Sequential DFS
    t1 = omp_get_wtime();
    dfs(0);
    t2 = omp_get_wtime();
    cout << "Time taken by Sequential DFS: "
         << t2 - t1 << " seconds" << endl;

    // Sequential BFS
    t1 = omp_get_wtime();
    bfs(0);
    t2 = omp_get_wtime();
    cout << "Time taken by Sequential BFS: "
         << t2 - t1 << " seconds" << endl;

    return 0;
}