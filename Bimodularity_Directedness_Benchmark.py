import numpy as np
from scipy.optimize import linear_sum_assignment

def adjacency_matrix_directedness_transform(adj_matrix, gamma):
    """
    Applies an element-wise directional strength transformation to an adjacency matrix.
    
    This function implements the formula:
    result[i,j] = 2 * A[i,j] * (A[i,j])^γ / ((A[i,j])^γ + (A[j,i])^γ)
    
    Each element is modified based only on its own value and the value of its 
    transpose counterpart, creating asymmetric directional emphasis. Higher gamma 
    values amplify existing edge strengths relative to their reverse directions.
    
    Parameters:
    -----------
    adj_matrix : numpy.ndarray
        Input adjacency matrix representing directed graph connections
    gamma : float
        Exponent parameter controlling directional emphasis (γ > 0)
        - γ = 1: Linear weighting based on forward/reverse edge ratio
        - γ > 1: Exponential amplification of stronger directions
        - γ < 1: Smoothing effect, reducing extreme directional differences
    
    Returns:
    --------
    numpy.ndarray
        Modified adjacency matrix where each element result[i,j] depends only on 
        the original elements A[i,j] and A[j,i]. All operations are element-wise.
        
    """
    # Convert to numpy array if not already
    A = adj_matrix.copy()

    A = np.array(A) ** gamma
    result = np.divide(A, A + A.T, out= np.zeros_like(A), where=(A + A.T) != 0)  # Avoid division by zero

    return result

def get_concat_binarized_bycommunities(sending_communities, receiving_communities):
    """
    Concatenates sending and receiving communities into a single binarized array.
    
    Parameters:
    -----------
    sending_communities : numpy.ndarray
        Array of sending community labels for each node
    receiving_communities : numpy.ndarray
        Array of receiving community labels for each node
        
    Returns:
    --------
    numpy.ndarray
        Concatenated array where each element is a tuple (sending, receiving) 
        representing the bicommunity assignment for each node.
        
    Notes:
    ------
    - The output will have the same length as the input arrays.
    - If the input arrays are not of equal length, an error will be raised.
    """

    bicommunities = np.concatenate((sending_communities, receiving_communities), axis=1)

    return (bicommunities > 0).astype(int)

def get_bicommunity_conjugate(bicommunities):
    """
    Computes the conjugate bicommunity assignment from a given bicommunity array.
    
    Parameters:
    -----------
    bicommunity : numpy.ndarray
        Array of bicommunity labels for each node (0 or 1)
        
    Returns:
    --------
    numpy.ndarray
        Conjugate bicommunity array where each element is the opposite of the 
        original (1 becomes 0, and 0 becomes 1).
        
    Notes:
    ------
    - This function is useful for generating complementary community assignments.
    """
    
    return  bicommunities ^ 1 

def cost_matrix_conjugate_bicommunity(conjugate_bicommunities, bicommunities):
    """
    Matches conjugate bicommunity assignments to original bicommunity labels.
    
    Parameters:
    -----------
    conjugate_bicommunity : numpy.ndarray
        Array of conjugate bicommunity labels (0 or 1)
    original_bicommunity : numpy.ndarray
        Array of original bicommunity labels (0 or 1)
        
    Returns:
    --------
    numpy.ndarray
        Array where each element indicates whether the conjugate matches the 
        original (1 for match, 0 for no match).
        
    Notes:
    ------
    - This function is useful for validating community assignments.
    """
    cost_matrix = []

    for conjugate_bicommunity in conjugate_bicommunities:
            distances = [np.sum((bicommunity ^ conjugate_bicommunity)) for bicommunity in bicommunities]
            cost_matrix.append(distances)
    return np.array(cost_matrix)

def community_fit_w_tolerance(sending_communities, receiving_communities,  epsilon = 0.5):
    """
    Computes the community fit score based on sending and receiving communities.
    
    Parameters:
    -----------
    sending_communities : numpy.ndarray
        Array of sending community labels for each node
    receiving_communities : numpy.ndarray
        Array of receiving community labels for each node
    bicommunities : numpy.ndarray
        Array of bicommunity labels for each node
        
    Returns:
    --------
    float
        Community fit score calculated as the sum of the absolute differences 
        between sending and receiving communities, normalized by the number of nodes.
        
    Notes:
    ------
    -   This function provides a measure of how well the communities align with 
        the bicommunity structure.
    """
    communities = sending_communities.shape[0] 
    nodes = sending_communities.shape[1] * 2 # each community has two columns, one for sending and one for receiving

    # values bigger than threshold will be considered as no match
    threshold =  nodes * (epsilon) # maximum hamming distance allowed

    # Concatenate sending and receiving communities into a single bicommunity array
    bicommunities = get_concat_binarized_bycommunities(sending_communities, receiving_communities)

    # Calculate the conjugate bicommunities
    conjugated_bicommunities = get_bicommunity_conjugate(bicommunities)

    # create a matrix cost with values punished for being outside tolerance, making more optimal to maximize the amount of matches within the tolerance before minimizing the cost
    # values outside tolerance  are discarded
    cost_matrix = cost_matrix_conjugate_bicommunity(conjugated_bicommunities, bicommunities) 
    cost_matrix[cost_matrix > threshold] = nodes**2 # sum of all distance is lesser than nodes**2

    # individual best matches, order is the same as in sending_communities
    singular_best_matches_indexes = (np.array([np.argmin(cost) for cost in cost_matrix]), np.arange(communities) )

    # remove matches outside tolerance
    mask = cost_matrix[singular_best_matches_indexes] < threshold
    singular_best_matches_indexes = singular_best_matches_indexes = (
        singular_best_matches_indexes[0][mask],
        singular_best_matches_indexes[1][mask]
    )  
    total_singular_cost = cost_matrix[singular_best_matches_indexes].sum()

    # overall_best_matches, order is the same as in sending_communities
    overall_best_matches_indexes = linear_sum_assignment(cost_matrix)  

    #remove matches outside tolerance
    mask = cost_matrix[overall_best_matches_indexes] < threshold
    overall_best_matches_indexes = overall_best_matches_indexes = (
        overall_best_matches_indexes[0][mask],
        overall_best_matches_indexes[1][mask]
    )  

    total_overall_cost = cost_matrix[overall_best_matches_indexes].sum()
    
    return singular_best_matches_indexes, total_singular_cost , overall_best_matches_indexes, total_overall_cost

