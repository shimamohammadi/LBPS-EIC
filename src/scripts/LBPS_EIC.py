import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scripts.tools import BT_model


class LBPS_EIC:

    def __init__(self, img_pairs, defer_num, delta=0.3, conditions=16):

        self.CONDITIONS = conditions
        self.defer_cc = 0
        self.delta = delta

        self.img_pairs = img_pairs.copy()
        self.normalize_un()
        self.current_pcm, self.pcm_un = self.get_pcm()
        self.defer_num = defer_num
        self.EIC = self.calculate_EIC()
        self.pairs_to_defer = self.select_pairs()

    def normalize_un(self):
        scaler = MinMaxScaler()
        self.img_pairs[['Model_Uncertainty']] = scaler.fit_transform(
            self.img_pairs[['Model_Uncertainty']])

    def kl_divergence_approx(self, mean_1, var_1, mean_2, var_2):
        '''
        Aproximation of the multivariate normal KL divergence: 
        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        '''
        total = np.sum(np.log(var_2)) - np.sum(np.log(var_1)) + \
            sum(var_1/var_2)+np.dot(1/var_2, (mean_1-mean_2)**2)
        return total

    def get_pcm(self):
        pcm_model = np.full((self.CONDITIONS, self.CONDITIONS), 0.00000001)
        pcm_un = np.zeros((self.CONDITIONS, self.CONDITIONS))
        idx_pcm = 0
        for row in range(self.CONDITIONS):
            for col in range(row+1, self.CONDITIONS):
                idx = self.img_pairs.index[idx_pcm]

                pcm_model[row, col] = self.img_pairs.loc[idx,
                                                         'Model_preference']
                pcm_model[col, row] = (
                    1 - self.img_pairs.loc[idx, 'Model_preference'])

                pcm_un[row, col] = self.img_pairs.loc[idx, 'Model_Uncertainty']
                pcm_un[col, row] = self.img_pairs.loc[idx, 'Model_Uncertainty']

                idx_pcm += 1

        return pcm_model, pcm_un

    def calculate_EIC(self):
        p_curr, pstd_curr = self.infer_scores(self.current_pcm)
        EIC = np.zeros((self.CONDITIONS, self.CONDITIONS))
        for row in range(self.CONDITIONS):
            for col in range(row+1, self.CONDITIONS):

                # Storing values of the current_pcm before updating their values
                temp_1 = self.current_pcm[row, col]
                temp_2 = self.current_pcm[col, row]

                self.current_pcm[row, col] = max(
                    0, min(1, (self.current_pcm[row, col]+max(self.pcm_un[row, col], self.delta))))

                self.current_pcm[col, row] = 1 - self.current_pcm[row, col]

                # Infer scores from the updated current_pcm
                p_tmp, pstd_tmp = self.infer_scores(self.current_pcm)

                # Calculate KLD between prior and posterior(current_pcm)
                kld_res1 = self.kl_divergence_approx(
                    p_curr, pstd_curr, p_tmp, pstd_tmp)

                # Recovering the current_pcm
                self.current_pcm[row, col] = temp_1
                self.current_pcm[col, row] = temp_2

                self.current_pcm[row, col] = max(
                    0, min(1, (self.current_pcm[row, col]-max(self.pcm_un[row, col], self.delta))))
                self.current_pcm[col, row] = 1 - self.current_pcm[row, col]

                # Infer scores from the updated current_pcm
                p_tmp, pstd_tmp = self.infer_scores(self.current_pcm)

                # Calculate KLD between prior and posterior(current_pcm)
                kld_res2 = self.kl_divergence_approx(
                    p_curr, pstd_curr, p_tmp, pstd_tmp)

                # Recovering the current_pcm
                self.current_pcm[row, col] = temp_1
                self.current_pcm[col, row] = temp_2

                kld_res = ((kld_res1-16) + (kld_res2-16))

                EIC[row, col] = ((kld_res))
                EIC[col, row] = ((kld_res))
        return EIC

    def infer_scores(self, pcm):
        """Infer scores from a PCM
        """
        _, _, _, p, pstd, _ = BT_model.BTM(pcm)
        return p, pstd

    def get_n_maximum_upper_triangle(self):

        if (self.defer_num == 0):
            return np.empty((0, 2), dtype=int)

        # Get the indices of the upper triangle of the matrix
        upper_tri_indices = np.triu_indices_from(self.EIC, k=1)

        # Get the values in the upper triangle
        upper_tri_values = self.EIC[upper_tri_indices]

        # Find the indices of the top n values in the upper triangle
        top_n_indices = np.argsort(upper_tri_values)[-self.defer_num:][::-1]

        # Get the corresponding row and column indices
        top_n_row_indices = upper_tri_indices[0][top_n_indices]
        top_n_col_indices = upper_tri_indices[1][top_n_indices]

        # Stack the row and column indices to form the result
        result = np.stack((top_n_row_indices, top_n_col_indices), axis=1)

        return result

    def select_pairs(self):

        pairs_to_defer = self.get_n_maximum_upper_triangle()

        return pairs_to_defer
