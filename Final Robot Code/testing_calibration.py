import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from mpl_toolkits.mplot3d import Axes3D

def test():
    num = 1
    for i in range(1):
        num = i + 1
        opti = np.genfromtxt(f"Fourth Year Project\\004 Programming\Linux code\opti_test_data_{num}", delimiter=",")
        robot = np.genfromtxt(f"Fourth Year Project\\004 Programming\Linux code\\robot_test_data_{num}", delimiter=",")
        
        
        for i in range(robot.shape[0]-1):
            # robot[i, 2] += -5
            opti_v = opti[i+1] - opti[i]
            robot_v = robot[i+1] - robot[i]
            robot_norm = np.linalg.norm(robot_v)
            opti_norm = np.linalg.norm(opti_v)
            print(abs(opti_norm- robot_norm))

        t, R, dists, avg_error, max_error = compute_matching_transform(opti, robot )

        print(np.linalg.det(R))



def find_transformation(A, B):

    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)

    A_centered = A- A_centroid
    B_centered = B- B_centroid

    H = B_centered.T @ A_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    t = A_centroid - R @ B_centroid

    return R, t


def compute_transform(source_points, target_points):
    """
    Compute the translation vector and rotation matrix between two sets of 3D points.
    """

    # Compute the centroids of the two sets of points
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Subtract the centroids to center the points around the origin
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Compute the rotation matrix using orthogonal Procrustes algorithm
    rotation_matrix, _ = orthogonal_procrustes(centered_source, centered_target)

    # Compute the translation vector
    translation_vector = centroid_target - np.dot(rotation_matrix, centroid_source)

    return translation_vector, rotation_matrix


def raw_test():

    # Define the vertices of the cube
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]])
    
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # Rotation matrix for 45 degrees about the x-axis
    Rx = np.array([[1, 0, 0],
                [0, np.sqrt(2)/2, -np.sqrt(2)/2],
                [0, np.sqrt(2)/2, np.sqrt(2)/2]])

    # Rotation matrix for 45 degrees about the y-axis
    Ry = np.array([[np.sqrt(2)/2, 0, np.sqrt(2)/2],
                [0, 1, 0],
                [-np.sqrt(2)/2, 0, np.sqrt(2)/2]])

    # Rotation matrix for 45 degrees about the z-axis
    Rz = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                [np.sqrt(2)/2, np.sqrt(2)/2, 0],
                [0, 0, 1]])
    
    vertices_new = vertices @ Rx @ Ry @ Rz
    vertices_new += np.ones(vertices_new.shape)
    vertices_new +=0.1*(np.random.standard_normal(vertices_new.shape))
    
    # t, R = compute_transform(vertices_new, vertices )
    # R, t = find_transformation(vertices, vertices_new )
    t, R, dists, avg_error, max_error = compute_matching_transform(vertices_new, vertices )
    # vertex_map = np.array([R @ (vertex_new - t) for vertex_new in vertices_new])
    vertex_map = np.array([t + R @ vertex_new for vertex_new in vertices_new])

    plot_cube(vertices, vertices_new, vertex_map, edges)


def plot_cube(vertices, vertices_2, vertices_3, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotting(ax, vertices, edges, 'tab:blue')
    plotting(ax, vertices_2, edges, 'tab:red')
    plotting(ax, vertices_3, edges, 'tab:purple')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cube')

    plt.show()

def plotting(ax, vertices, edges, color):
    # Plot the vertices
    for i, vertex in enumerate(vertices):
        ax.scatter(vertex[0], vertex[1], vertex[2], color=color)
        ax.text(vertex[0], vertex[1], vertex[2], f"{i}", color=color)

    for edge in edges:
        ax.plot3D([vertices[edge[0]][0], vertices[edge[1]][0]],
                  [vertices[edge[0]][1], vertices[edge[1]][1]],
                  [vertices[edge[0]][2], vertices[edge[1]][2]], color=color)
        

def compute_matching_transform(pointsA, pointsB):
    assert len(pointsA) == len(pointsB)
    N = len(pointsA)
    comA = np.sum(pointsA, axis=0) / len(pointsA)
    comB = np.sum(pointsB, axis=0) / len(pointsB)

    p_A = [p - comA for p in pointsA]
    p_B = [p - comB for p in pointsB]

    H = np.zeros((3, 3))
    for i in range(N):
        H += np.outer(p_A[i], p_B[i])

    U, _, V = np.linalg.svd(H)
    # Flip direction of least significant vector to turn into a rotation if a reflection is found.
    # Required when the points are all coplanar.
    V = V.T @ np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, np.linalg.det(V) * np.linalg.det(U)]])
    R = V @ U.T
    # Double check
    # assert np.allclose(R @ comA, comB - R @ comA)
    assert np.isclose(np.linalg.det(R), 1)

    translation = comB - R @ comA

    # Compute errors
    errs = [pointsB[i] - (translation + R @ pointsA[i]) for i in range(N)]
    dists = [np.linalg.norm(err) for err in errs]
    avg_error = sum(dists) / len(dists)
    max_error = max(dists)
    

    print(f"Average error for each point: {1e3 * avg_error}mm. Maximum error (worst point): {1e3 * max_error}mm")
    return translation, R, dists, avg_error, max_error

def integral_plot():
    robot_poses = np.genfromtxt("integral testing.csv", delimiter=",")
    goal_point = np.array([ 0.36457688, -0.04057294,  0.36632183])
    robot_poses[:,3] *=1e9
    times = (robot_poses[:,3]+ robot_poses[:,4]) - (robot_poses[0,3]+robot_poses[0,4])
    times /= 1e9
    robot_poses = robot_poses[:,:3]
    error = goal_point - robot_poses
    labels= ["x", "y", "z"]
    plt.plot(times, error, label=labels)
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.show()


# raw_test()
integral_plot()
