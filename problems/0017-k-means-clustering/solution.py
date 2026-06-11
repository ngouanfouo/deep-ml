import math
import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    
    def euclidean_distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def compute_mean(points_list):
        """Compute the mean (centroid) of a list of points."""
        if not points_list:
            return None
        num_points = len(points_list)
        # Sum coordinates element-wise
        sums = [sum(coords) for coords in zip(*points_list)]
        # Divide by number of points
        mean = tuple(s / num_points for s in sums)
        return mean
    
    # Convert initial centroids to list format
    centroids = list(initial_centroids)
    
    for iteration in range(max_iterations):
        # Step 1: Assign each point to the nearest centroid
        clusters = [[] for _ in range(k)]
        
        for point in points:
            # Find the closest centroid
            min_distance = float('inf')
            closest_cluster = 0
            
            for i, centroid in enumerate(centroids):
                dist = euclidean_distance(point, centroid)
                if dist < min_distance:
                    min_distance = dist
                    closest_cluster = i
            
            clusters[closest_cluster].append(point)
        
        # Step 2: Update centroids based on cluster assignments
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                # Compute mean of points in this cluster
                new_centroid = compute_mean(clusters[i])
                new_centroids.append(new_centroid)
            else:
                # If cluster is empty, keep the old centroid
                new_centroids.append(centroids[i])
        
        # Step 3: Check for convergence
        # If centroids haven't changed, we're done
        if new_centroids == centroids:
            break
            
        centroids = new_centroids
    
    # Round final centroids to 4 decimal places and ensure tuple format
    final_centroids = []
    for centroid in centroids:
        rounded_centroid = tuple(round(coord, 4) for coord in centroid)
        final_centroids.append(rounded_centroid)
    
    return final_centroids