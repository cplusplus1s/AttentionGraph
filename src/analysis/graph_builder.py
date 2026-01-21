import numpy as np
import networkx as nx

class GraphBuilder:
    def build_signal_graph(self, matrix, names, threshold_std=2.0):
        threshold = np.mean(matrix) + threshold_std * np.std(matrix)
        G = nx.DiGraph()
        n = len(names)

        for i in range(n): # Target
            for j in range(n): # Source
                if i == j: continue
                weight = matrix[i, j]
                if weight > threshold:
                    G.add_edge(names[j], names[i], weight=weight)
        return G

    def build_module_graph(self, matrix, names, mapping_dict, threshold_offset=0.4):
        # 1. Mapping names to modules
        module_names = []
        for name in names:
            found = False
            for key, val in mapping_dict.items():
                if key in name:
                    module_names.append(val)
                    found = True
                    break
            if not found:
                module_names.append(mapping_dict.get("Default", "Other"))

        unique_modules = sorted(list(set(module_names)))
        n_mod = len(unique_modules)

        # 2. Aggregation matrix
        mod_matrix = np.zeros((n_mod, n_mod))
        count_matrix = np.zeros((n_mod, n_mod))

        for i in range(len(names)): # Receiver
            for j in range(len(names)): # Source
                r_idx = unique_modules.index(module_names[i])
                c_idx = unique_modules.index(module_names[j])

                mod_matrix[r_idx, c_idx] += matrix[i, j]
                count_matrix[r_idx, c_idx] += 1

        # Avoid dividing by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_mod_matrix = np.nan_to_num(mod_matrix / count_matrix)

        # 3. Plot
        G_mod = nx.DiGraph()
        threshold = np.mean(avg_mod_matrix) + threshold_offset * np.std(avg_mod_matrix)

        for r in range(n_mod):
            for c in range(n_mod):
                weight = avg_mod_matrix[r, c]
                # Keep all nodes, filter edges
                G_mod.add_node(unique_modules[r])
                if weight > threshold:
                    G_mod.add_edge(unique_modules[c], unique_modules[r], weight=weight)

        return G_mod