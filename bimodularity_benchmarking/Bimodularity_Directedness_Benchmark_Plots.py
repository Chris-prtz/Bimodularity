import numpy as np
import Bimodularity_Directedness_Benchmark as bimod_bench
import matplotlib.pyplot as plt
import dgsp
import seaborn as sns

def  get_communities(A_sym, A_assym, alphas, gammas, n_kmeans, vector_id_max): 
    communities = []
    for gamma in gammas:
        g_com = []
        for alpha in alphas:
            model =  bimod_bench.adjacency_matrix_directedness_transform(A_sym + alpha*A_assym, gamma = gamma)

            # determine bicomunities 
            U, S, Vh = dgsp.sorted_SVD(dgsp.modularity_matrix(model, null_model="outin"))
            V = Vh.T

            edge_clusters, edge_clusters_mat = dgsp.edge_bicommunities(model, U, V, vector_id_max, method="kmeans",
                                                                    n_kmeans=n_kmeans, scale_S= S[:vector_id_max], verbose=False)
            n_clusters = np.max(edge_clusters)

            sending_communities, receiving_communities = dgsp.get_node_clusters(edge_clusters, edge_clusters_mat, method="bimodularity")  
            g_com.append((sending_communities, receiving_communities))
        communities.append(g_com)
    return communities

    

def alpha_gamma_heatmaps(communities, tolerance_range, n_gamma_values, n_alpha_values, ):
    """
    Generates heatmaps for community matches at different tolerance levels.
    """

    heatmaps_singular = []
    heatmaps_overall = []

    for tolerance in tolerance_range: 
        singular_match_counts = np.zeros((n_gamma_values, n_alpha_values))
        overall_match_counts = np.zeros((n_gamma_values, n_alpha_values))
        
        for i, g_com in enumerate(communities): 
            for j, (sending_communities, receiving_communities) in enumerate(g_com):
                fit_result = bimod_bench.community_fit_w_tolerance(sending_communities, receiving_communities, tolerance)
                
                # Extract surviving indices
                singular_surviving_indices = fit_result[0][1]  # singular community match indices
                overall_surviving_indices = fit_result[2][1]   # overall community match indices
                
                # Store counts as before
                singular_match_counts[i, j] = len(singular_surviving_indices)
                overall_match_counts[i, j] = len(overall_surviving_indices)
                
        heatmaps_singular.append(singular_match_counts)
        heatmaps_overall.append(overall_match_counts)

    return heatmaps_singular, heatmaps_overall

def plot_heatmaps( alphas, gammas, n_kmeans, tolerance_range, heatmaps, title):
    """
    Plots heatmaps for community matches at different tolerance levels.
    """
    n_tolerance = len(tolerance_range)
    n_rows = int(np.ceil(n_tolerance / 5))

    # Calculate global min/max for consistent color scaling
    v_min, v_max = 0, n_kmeans
    # Add tick labels to all plots
    n_ticks = 6
    alpha_tick_indices = np.linspace(0, len(alphas) -1, n_ticks, dtype=int)
    gamma_tick_indices = np.linspace(0, len(gammas) -1, n_ticks, dtype=int)
    alpha_tick_labels = [f'{alphas[i]:.1f}' for i in alpha_tick_indices]
    gamma_tick_labels = [f'{gammas[i]:.1f}' for i in gamma_tick_indices]


    # Make each heatmap bigger with square cells
    fig, axes = plt.subplots( n_rows , 5, figsize=(15, n_rows * 3),)
    fig.suptitle(title,fontsize=14, y=0.98)
    plt.subplots_adjust(right=0.85)


    for i in range(n_rows):
        for j in range(5):
            tolerance = tolerance_range[5*i + j] 
            data_to_plot = heatmaps[ 5*i + j]
            im = sns.heatmap(data_to_plot, ax=axes[i, j], cmap='viridis', cbar=False,
                                vmin = v_min, vmax= v_max,
                                square=True, annot=False, fmt='d' )
            axes[i, j].set_xticks (alpha_tick_indices, alpha_tick_labels)
            axes[i, j].set_xlabel('Alpha', fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel('Gamma', fontsize=10)

            axes[i, j].set_yticks(gamma_tick_indices, gamma_tick_labels)


            axes[i, j].set_title(f'Tolerance: {tolerance:.2f}', fontsize=10, fontweight='bold')
    
    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im.collections[0], cax=cbar_ax)
    cbar.set_label('Number of Communities', rotation=270, labelpad=20)



def show_repartition(communities, tolerance, alphas, gammas, k):
    """
    Shows the repartition of communities surviving at different tolerances.
    """
    counts = bimod_bench.get_surviving_communities(communities, tolerance, k, alphas, gammas)
    print("Counts of surviving communities at tolerance", tolerance)
    print(counts)