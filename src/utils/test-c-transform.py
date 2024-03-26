import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# read the label_roots_vr_18_SNR_3_res_256x256x191.raw
label = np.fromfile(
    "../data/label_roots_vr_18_SNR_3_res_256x256x131.raw", dtype="int16"
)
label = label.reshape((131, 256, 256))

# read the III_Sand_1W_DAP14_256x256x191.raw
image = np.fromfile("../data/III_Sand_1W_DAP14_256x256x131.raw", dtype="int16")
image = image.reshape((131, 256, 256))
# set all values of image which are greater than 99.99% percentile to 1
image = np.where(image > np.percentile(image, 99.99), 1, 0)

label_points = np.argwhere(label > 0)
image_points = np.argwhere(image > 0)

label_points = label_points.astype(np.float64)
image_points = image_points.astype(np.float64)

print("label_points.shape:", label_points.shape)

label_point_cloud = o3d.geometry.PointCloud()
label_point_cloud.points = o3d.utility.Vector3dVector(label_points)

image_point_cloud = o3d.geometry.PointCloud()
image_point_cloud.points = o3d.utility.Vector3dVector(image_points)


# calculate features
def calc_features(point_cloud):
    # Estimate normals
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30)
    )
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=100)
    )

    return fpfh


label_features = calc_features(label_point_cloud)
image_features = calc_features(image_point_cloud)

print("label_features.data.shape:", label_features.data.shape)
print("image_features.data.shape:", image_features.data.shape)

# Convert FPFH features to numpy arrays
fpfh_1 = np.asarray(label_features.data).T
fpfh_2 = np.asarray(image_features.data).T

# Create KD-Trees
tree_1 = cKDTree(fpfh_1)
tree_2 = cKDTree(fpfh_2)

# Find nearest neighbors (you can use k=2 for Lowe's ratio test)
distances, indices = tree_2.query(fpfh_1, k=1)

# plt.hist(distances, bins=80)
# plt.xlabel("Distance")
# plt.ylabel("Frequency")
# plt.title("Distribution of Feature Distances")
# plt.show()


# Filter matches (example: distance threshold)
threshold = 40  # Adjust based on your dataset
good_matches = [
    (i, indices[i]) for i in range(len(indices)) if distances[i] < threshold
]
good_matches_array = np.array(good_matches, dtype=np.int32)
print("good_matches_array.shape:", good_matches_array.shape)

print("good_matches:", len(good_matches))


source_points = np.array([label_point_cloud.points[idx] for idx, _ in good_matches])
target_points = np.array([image_point_cloud.points[idx] for _, idx in good_matches])

# Convert to Open3D point clouds for RANSAC
source = o3d.geometry.PointCloud()
target = o3d.geometry.PointCloud()

source.points = o3d.utility.Vector3dVector(source_points)
target.points = o3d.utility.Vector3dVector(target_points)

# Run RANSAC
threshold = 0.05  # Set a threshold for RANSAC, adjust according to your data
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    source,
    target,
    o3d.utility.Vector2iVector(array_2d=good_matches),
    threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
        False
    ),
    ransac_n=4,
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
)

# The transformation matrix
transformation_matrix = ransac_result.transformation

print("Transformation Matrix:", transformation_matrix)
