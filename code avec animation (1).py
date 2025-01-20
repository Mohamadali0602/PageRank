
import numpy.random as rd
import numpy as np
import scipy.sparse as sp
import random
import matplotlib.pyplot as plt
import time
# ==============================
# 1) Génération et normalisation
# ==============================

def generate_adjacency_matrix(n=10, p=0.5):
    """
    Génère une matrice d’adjacence binaire n x n :
    - Avec probabilité p, on met un lien (i->j).
    - On force au moins un lien sortant pour chaque i.
    """
    row = []
    col = []
    data = []
    for i in range(n):
        count_out = 0
        for j in range(n):
            if np.random.binomial(1, p): # pour que la matrice soit aléatoire, on utilise la loi de bernoulli pour avoir le coeff i,j
                row.append(i)
                col.append(j)
                data.append(1)
                count_out += 1
        if count_out == 0:  # On Assure qu'il y a au moins un lien sortant
            target = random.randint(0, n - 1)
            row.append(i)
            col.append(target)
            data.append(1)

    A = sp.csr_matrix((data, (row, col)), shape=(n, n)) 
    # on stocke les données de la matrice creuse pour pouvoir effectué des calculs pour n grand
    return A

def create_transition_matrix(A):
    """
    Transforme une matrice d’adjacence brute A en matrice stochastique M.
    """
    M = A.multiply(1 / A.sum(axis=1))  # On normalise chaque ligne
    return M

# =====================
# 2) PageRank et Naïve
# =====================

def google_matrix(M, alpha=0.85):
    """
    Construit la matrice Google G = alpha*M + (1 - alpha)*R,
    où R est la matrice 1/n partout (row-stochastic).
    """
    n = M.shape[0]
    R = sp.csr_matrix(np.ones((n, n)) / n) 
    G = alpha * M + (1 - alpha) * R # Formule de la matrice de page rank
    return G

def pagerank_iterative(G, tol=1e-6, max_iter=1000):
    """
    Calcule le PageRank en itérant sur G (matrice stochastique).
    Retourne (vecteur pagerank, nb_iterations_effectives).
    """
    n = G.shape[0]
    rank = np.ones(n) / n
    for i in range(1, max_iter + 1):
        new_rank = rank @ G
        if np.linalg.norm(new_rank - rank, 1) < tol:
            return new_rank, i
        rank = new_rank
    return rank, max_iter

def naive_method(A):
    """
    Calcule le score basé sur la méthode naïve (somme des liens entrants).
    """
    in_degrees = np.array(A.sum(axis=0)).flatten()
    return in_degrees / np.sum(in_degrees)  # Onormalisation pour obtenir des proportions

# ============================
# 3) Surfeur aléatoire avec compteur
# ============================

def random_surfer_all_pages(M, num_steps=1000, top_page_idx=None, second_top_page_idx=None):
    """
    Simule le déplacement d'un surfeur aléatoire sur M à partir de toutes les pages.
    Compte combien de fois il passe sur les deux pages les plus importantes.
    """
    n = M.shape[0]
    total_visits_top = 0  # Compteur global pour la page la plus importante
    total_visits_second = 0  # Compteur global pour la deuxième page la plus importante

    for start_page in range(n):  # Parcourir toutes les pages comme points de départ
        current_page = start_page
        visits_top = 0  # Compteur local pour la page top
        visits_second = 0  # Compteur local pour la deuxième page top

        for step in range(num_steps):
            # Probabilités pour la prochaine page
            row_probs = M.getrow(current_page).toarray().ravel()
            next_page = np.random.choice(n, p=row_probs)

            # Incrémentation des compteurs
            if next_page == top_page_idx:
                visits_top += 1
            if next_page == second_top_page_idx:
                visits_second += 1

            current_page = next_page

        print(f"Départ depuis la page {start_page+1} : {visits_top} visites sur la page top (#{top_page_idx+1}), {visits_second} visites sur la deuxième page (#{second_top_page_idx+1}).")
        total_visits_top += visits_top
        total_visits_second += visits_second

    print(f"\nTotal des visites du surfeur sur la page top (#{top_page_idx+1}) : {total_visits_top} pour {n} départs.")
    print(f"Total des visites du surfeur sur la deuxième page (#{second_top_page_idx+1}) : {total_visits_second} pour {n} départs.")
    return total_visits_top, total_visits_second

# ============================
# 4) Visualisation
# ============================

def plot_comparison_graph(A, pagerank_vals,top_pagerank_idx, top_naive_idx):
    """
    Affiche un graphe des pages avec la page la plus importante selon :
    - PageRank (rouge)
    - Méthode Naïve (vert)
    """
    n = A.shape[0]
    x = np.random.rand(n)
    y = np.random.rand(n)

    row_idx, col_idx = A.nonzero()  # Obtenir les indices des liens
    for i, j in zip(row_idx, col_idx):
        plt.plot([x[i], x[j]], [y[i], y[j]], color='gray', alpha=0.5, lw=0.5)

    sizes = [pagerank_vals[i] * 800 / max(pagerank_vals) for i in range(n)]
    colors = ['blue'] * n
    colors[top_pagerank_idx] = 'red'  # Page importante selon PageRank
    colors[top_naive_idx] = 'green'  # Page importante selon méthode naïve

    plt.scatter(x, y, s=sizes, c=colors, alpha=0.8)
    for i in range(n):
        plt.text(x[i], y[i], str(i+1), fontsize=15, ha='center', va='center', color='white')

    plt.title("Graphe : PageRank vs Méthode Naïve\n(Rouge = PageRank, Vert = Méthode Naïve)")
    plt.axis('off')
    plt.show()


# =====================
# 5) Main
# =====================
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    n = 10 # nb de page
    p = 0.5 # proba d aller d'une page à une autre  
    alpha = 0.85 # alpha qui représente l'importance de la matrice stochastique originale.
    #Plus alpha est grand, moins on prend en compte la matrice R de teleportation 
    max_iter = 1000 #nombre d'iterations
    tol = 1e-6 # erreur
    num_steps_for_surfer = 1000 
    actual_page = 1 # la page de départ 

    A = generate_adjacency_matrix(n, p) #on genere une matrice
    M = create_transition_matrix(A) # on la rend stochastique
    G = google_matrix(M, alpha) # on la transforme en matrice de google

    pagerank_vals, iters = pagerank_iterative(G, tol=tol, max_iter=max_iter)
    print(f"PageRank a convergé en {iters} itérations (sur {max_iter} max).")

    top_pagerank_idx = np.argsort(pagerank_vals)[-1]
    second_top_pagerank_idx = np.argsort(pagerank_vals)[-2]

    print(f"Page la plus importante (PageRank) : Page {top_pagerank_idx + 1}")
    print(f"Deuxième page la plus importante (PageRank) : Page {second_top_pagerank_idx + 1}")

    naive_vals = naive_method(A)
    top_naive_idx = np.argsort(naive_vals)[-1]
    second_top_naive_idx = np.argsort(naive_vals)[-2]

    print(f"Page la plus importante (Naïve) : Page {top_naive_idx + 1}")
    print(f"Deuxième page la plus importante (Naïve) : Page {second_top_naive_idx + 1}")

    print("\n--- Surfeur aléatoire (départs depuis toutes les pages) ---")
    random_surfer_all_pages(M, num_steps=num_steps_for_surfer, top_page_idx=top_pagerank_idx, second_top_page_idx=second_top_pagerank_idx)

    # Animation

    pagerank_vals, iters = pagerank_iterative(G, tol=tol, max_iter=max_iter) 
    print(f"PageRank a convergé en {iters} itérations (sur {max_iter} max).")

    top_pagerank_idx = np.argsort(pagerank_vals)[-1]
    second_top_pagerank_idx = np.argsort(pagerank_vals)[-2]

    naive_vals = naive_method(A)
    top_naive_idx = np.argsort(naive_vals)[-1]

    x = np.random.rand(n)
    y = np.random.rand(n)

    row_idx, col_idx = A.nonzero() 
    sizes = [pagerank_vals[i] * 800 / max(pagerank_vals) for i in range(n)] 

    plt.ion() 
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, j in zip(row_idx, col_idx):
        ax.plot([x[i], x[j]], [y[i], y[j]], color='gray', alpha=0.5, lw=0.5)

    colors = ['blue'] * n
    scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.8)

    for i in range(n):
        ax.text(x[i], y[i], str(i + 1), fontsize=15, ha='center', va='center', color='white')

    ax.set_title("Graphe Dynamique : Surfeur Aléatoire")
    ax.axis('off')

    color_change_counts = [0] * n

    text_box = ax.text(1.05, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=10)

    for z in range(max_iter):  
        if z < 10: 
            time.sleep(1)
        probabilites = G[actual_page, :].toarray().flatten()
        probabilites /= probabilites.sum()
        next_page_pagerank = np.random.choice(np.arange(n), p=probabilites)
        actual_page = next_page_pagerank

        colors = ['blue'] * n
        colors[actual_page] = 'red'  
        scatter.set_color(colors)

      
        color_change_counts[actual_page] += 1

     
        counts_text = "\n".join([f"Node {i + 1}: {count}" for i, count in enumerate(color_change_counts)])
        text_box.set_text(counts_text)

        plt.pause(0.01)  

  
    plt.ioff()  

    display_final_plot(x, y, row_idx, col_idx, sizes, pagerank_vals, naive_vals, top_pagerank_idx, top_naive_idx, color_change_counts)


def display_final_plot(x, y, row_idx, col_idx, sizes, pagerank_vals, naive_vals, top_pagerank_idx, top_naive_idx, color_change_counts):
    plt.figure(figsize=(10, 8))  
    ax_final = plt.gca()

    for i, j in zip(row_idx, col_idx):
        ax_final.plot([x[i], x[j]], [y[i], y[j]], color='gray', alpha=0.5, lw=0.5)


    final_colors = ['blue'] * len(x)
    final_colors[top_pagerank_idx] = 'red'  
    final_colors[top_naive_idx] = 'green' 

    ax_final.scatter(x, y, s=sizes, c=final_colors, alpha=0.8)

    for i in range(len(x)):
        label = f"{pagerank_vals[i]:.2f}\n{naive_vals[i]:.2f}"
        ax_final.text(x[i], y[i], label, fontsize=9, ha='center', va='center', color='white')

    ax_final.set_title(
        "Final Graphe : PageRank et Méthode Naïve\n"
        "(Rouge = PageRank, Vert = Méthode Naïve)\n"
        "(Node: PageRank / Naïve)"
    )
    ax_final.axis('off')

    counts_text = "\n".join([f"Node {i + 1}: {count}" for i, count in enumerate(color_change_counts)])

    ax_final.text(1.05, 0.95, counts_text, transform=ax_final.transAxes, ha="left", va="top", fontsize=10)

    plt.show()


if __name__ == "__main__":
    main()