### Clustering
    clustering is the unsupervised machine learning technique. it groups the data num of clusters based on the clustering alogirthm

### Cluster Algorithm Types
1. kmeans - K-Means Clustering
2. ap  -  Affinity Propagation
3. meanshift - Mean Shift Clustering
4. sc - Spectral Clustering
5. hclust - Agglomerative Clustering
6. dbscan - Density-Based Spatial Clustering
7. optics - OPTICS Clustering
8. birch - Birch Clustering

## kmeans - K-Means Clustering
    step 1: we randomly select the k num of points in the datas
    
    step 2: calculate the distance (euclidence distance) between the each points 

    k = 3    
    formula = √(a2+b2+c3)

    step 3: calculate the mean of the each data points (sum of terms / num of terms)

    step 4: depends upon the variations(k) it will clusters that. 

    note: the main point of this clusters it depends on the k value so we need to check the perfect clusters based on the k value. 

## ap  -  Affinity Propagation
    step 1 - ap creates the exempler, points that explains the other data points and are the most significant of their clusters.  

    step 2 - the algorithm does not require the number of the predetermind clusters (k)

    step 1 (similarity) - as an input, the algorithm requires us to provide 2 sets of data:
        the basic idea behind ap is to represent each data point as a node in a n/w and to define a similarity metric b/w nodes. the similarity metric measures the degree to which 2 data points are similar or dis-similar.
        it can be using various measures concepts 
        1. euclidean distance 
        2. cosine similarity

## meanshift - Mean Shift Clustering
    mean of each datapoints shift its one particular group that is the meanshift. it work arround kernal density estimation (KDE)
    based on the kernal density it will calculate the clustering

## sc - Spectral Clustering
    it split the data as 2 category then check the nearby values that areas 

## hclust - Agglomerative Clustering
    it is called as the hierarchial clustering using the agglomerative (bottom up approch) approch.

    the 2 data points are hierachy it gives the another one point it is the added the cluster group.

    step 1: using the proximity measures to calculate the cluster

    proximity measure:
        similarity, dis-similarity 2 objects 2 different cluster.
        
        using the euclidean distance, manhaltan distance
    merge the all of the sub clusters to the parent clusters 

## dbscan - Density-Based Spatial Clustering
    clustering algorithm that can identify nested clusters in high-dimensions(multiple axis x, y, z,etc..). 

    step 1: identifies the clusters based on the density.

    step 2: choose the randomly any data points and check the closer to how much points 

    step 3: check the corepoint to others then extended that finally it included the all of the overlap mapping core point add it a cluster group

    step 4: non corepoints are called the outlier its not included any group of clusters

    Note: The number of close points for a CORE POINT is user defined, so when using DBSCAN, you might need to fiddle with this parameters as well.


## optics - OPTICS Clustering
    optics stands for ordering point to identify cluster structure. density based cluster algorithm.
    
    step 1: core distance - min value of radius required to classify a given point as a core point.

    step 2: Reachability distance - It is defined with respect to another data point q(Let). The Reachability distance between a point p and q is the maximum of the Core Distance of p and the Euclidean Distance(or some other distance metric) between p and q. Note that The Reachability Distance is not defined if q is not a Core point. technique is different from other clustering techniques in the sense that this technique does not explicitly segment the data into clusters. Instead, it produces a visualization of Reachability distances and uses this visualization to cluster the data. 
    
## birch - Birch Clustering

## tested clusters report

1. kmeans - K-Means Clustering - work
2. ap  -  Affinity Propagation - work
3. meanshift - Mean Shift Clustering - work
4. sc - Spectral Clustering - not work
5. hclust - Agglomerative Clustering - not work
6. dbscan - Density-Based Spatial Clustering - not work
7. optics - OPTICS Clustering - not work
8. birch - Birch Clustering - work
