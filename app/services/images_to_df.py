import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import sobel

class ImagesToDf(object):
    def __init__(self):
        self.gabor_kernels = self.create_gabor_kernels()
        print(f"Gabor kernel size: {len(self.gabor_kernels)}")

    @staticmethod
    def dataframe_merge_columns(df1, df2):
        res = pd.merge(df1, df2, left_index=True, right_index=True)
        return res

    @staticmethod
    def dataframe_merge_rows(df1, df2):
        res = pd.concat([df1, df2])
        return res

    @staticmethod
    def create_gabor_kernels():
        """
        :description: Generate Gabor features
        :return: kernel list
        """
        kernels = []  # Create empty list to hold all kernels that we will generate in a loop
        for theta in range(2):  # Define number of thetas. Here only 2 theta values 0 and 1/4 . pi
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # Sigma with values of 1 and 3
                for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                    for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                        ksize = 3
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        kernels.append(kernel)
        return kernels

    @staticmethod
    def remove_null_rows(df):
        # Az első és utolsó oszlopot különválasztjuk
        first_col = df.iloc[:, 0]
        last_col = df.iloc[:, -1]
        # Köztes oszlopok kiválasztása
        middle_cols = df.iloc[:, 1:-1]
        # Feltétel, hogy a köztes oszlopokban csak nulla vagy üres értékek legyenek
        condition = middle_cols.map(lambda x: x == 0 or pd.isna(x)).all(axis=1)
        # Sorok, amelyek nem felelnek meg a feltételnek
        df_cleaned = df[~condition]
        return df_cleaned

    @staticmethod
    def select_random_rows(df, selected_values):
        filtered_dfs = []
        for value in selected_values:
            # Kiválasztjuk a Label oszlopban a value-hoz tartozó sorokat
            temp_df = df[df['Label'] == value]
            # Véletlenszerűen kiválasztunk 100000 sort (ha kevesebb, akkor annyit amennyi elérhető)
            sampled_df = temp_df.sample(n=min(1000, len(temp_df)), random_state=99)
            filtered_dfs.append(sampled_df)

        df = df[~df['Label'].isin(selected_values)]

        # Az új dataframe létrehozása, az eredeti sorrend megtartásával
        new_df = pd.concat(filtered_dfs).sort_index()
        return new_df  # dataframe_merge_rows(df, new_df)

    def multi_filter(self, patient_id, img, tumor_type, lung_state):
        """
        Minden pixelre megvan hívva ez a metódus
        Gabor, SOBEL, GAUSSIAN with sigma=3, GAUSSIAN with sigma=7, MEDIAN with sigma=3, VARIANCE with size=3 filters
        label_number: 1: egészséges tüdő, 2 beteg tüdő, 3: egészséges lágyszövet, 4: betet lágyszövet
        :param Patient ID
        :param img:
        :param tumor_type: tumor type
        :param lung_state: diseased_lungs, healthy_lungs, diseased_soft_tissue, healthy_soft_tissue
        :return: Pandas Dataframe
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        df = pd.DataFrame()
        img2 = img.reshape(-1)
        df["Image"] = img2

        num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
        for gabor in self.gabor_kernels:
            gabor_label = 'Gabor' + str(num)
            fimg = cv2.filter2D(img2.astype('float32'), cv2.CV_32F, gabor)
            filtered_img = fimg.reshape(-1)
            df[gabor_label] = filtered_img
            num += 1

        # SOBEL
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1

        # GAUSSIAN with sigma=3
        from scipy import ndimage as nd
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)
        df['Gaussian_s3'] = gaussian_img1

        # GAUSSIAN with sigma=7
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img3 = gaussian_img2.reshape(-1)
        df['Gaussian_s7'] = gaussian_img3

        # MEDIAN with sigma=3
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)
        df['Median_s3'] = median_img1

        # VARIANCE with size=3
        variance_img = nd.generic_filter(img, np.var, size=3)
        variance_img1 = variance_img.reshape(-1)
        df['Variance_s3'] = variance_img1  # Add column to original dataframe

        if lung_state == "healthy_lungs":  # egészséges tüdő (ki szegmentált)
            label_value = 1
        elif lung_state == "diseased_lungs":  # beteg tüdő (teljes kép)
            if tumor_type == 'A':
                label_value = 4
            elif tumor_type == 'B':
                label_value = 5
            elif tumor_type == 'D':
                label_value = 6
            elif tumor_type == 'G':
                label_value = 7
            else:
                label_value = 0
        elif lung_state == "healthy_soft_tissue":  # egészséges lágyszövet (tumor mellett) ROI
            label_value = 1
        elif lung_state == "diseased_soft_tissue":  # beteg lágyszövet (szegmentált tumor) ROI
            if tumor_type == 'A':
                label_value = 8
            elif tumor_type == 'B':
                label_value = 10
            elif tumor_type == 'D':
                label_value = 12
            elif tumor_type == 'G':
                label_value = 14
            else:  # Ha, fekete(háttér pixel)
                label_value = 0
        else:
            label_value = 0
        df["Label"] = label_value
        df["patient_id"] = patient_id

        return df

    def preproccessing_images(self, four_img_array):
        """
                Run the Gabor, SOBEL, GAUSSIAN with sigma=3, GAUSSIAN with sigma=7, MEDIAN with sigma=3,
                VARIANCE with size=3 filters on the images of four classes, and then convert them into a column vector.
                Fusion of received pixel rows of images.
                Minden szelet négy osztály képére lefutattni a Gabor, SOBEL, GAUSSIAN with sigma=3, GAUSSIAN with sigma=7,
                MEDIAN with sigma=3, VARIANCE with size=3 szűröket.majd ezek oszlop vektorrá alakítása.
                A sorokhoz hozzárendel egy besorolási számot. Ez Label sorban van.
                Képek kapott pixel sorainak egymás alá fúzése.
                :param four_img_array:
                [0]: eredeti kép tömbje,
                [1]: szegmentált egészséges tüdő (parenchyma),
                [2]: szegmentált tumor,
                [3]: szegmentált tumor körüli egészséges tüdő,
                [4]: A vagy B vagy D vagy G,
                [5]: páciens neve
                :return: Pandas Dataframe
                """
        df_all = None
        slice_counter = 1
        for i in tqdm(range(len(four_img_array)), desc="Preproccessing images"):
            # beteg tüdő(egész kép)
            diseased_lungs_df = self.multi_filter(four_img_array[i][5], four_img_array[i][0], four_img_array[i][4],
                                                  lung_state="diseased_lungs")
            # Töröljük a teljesen üres sorokat.
            diseased_lungs_df = self.remove_null_rows(diseased_lungs_df)
            # Kiválasztjuk azokat a sorokat, ahol a 'Label' oszlop értéke 4, 5, 6 vagy 7
            diseased_lungs_df = self.select_random_rows(diseased_lungs_df, [0, 4, 5, 6, 7])
            # egészséges tüdő (szegmentált tüdő)
            # multi_filter(páciens neve, egészséges/beteg tüdő/, A vagy B vagy D vagy G,  )
            healthy_lungs_df = self.multi_filter(four_img_array[i][5], four_img_array[i][1], four_img_array[i][4],
                                                 lung_state="healthy_lungs")
            # Töröljük a teljesen üres sorokat.
            healthy_lungs_df = self.remove_null_rows(healthy_lungs_df)
            # Kiválasztjuk azokat a sorokat, ahol a 'Label' oszlop értéke 1
            healthy_lungs_df = self.select_random_rows(healthy_lungs_df, [0, 1])
            # beteg lágyszövet
            diseased_soft_tissue_df = self.multi_filter(four_img_array[i][5], four_img_array[i][2],
                                                        four_img_array[i][4],
                                                        lung_state="diseased_soft_tissue")
            # Töröljük a teljesen üres sorokat.
            diseased_soft_tissue_df = self.remove_null_rows(diseased_soft_tissue_df)
            # Kiválasztjuk azokat a sorokat, ahol a 'Label' oszlop értéke 8, 10, 12 vagy 14
            diseased_soft_tissue_df = self.select_random_rows(diseased_soft_tissue_df, [0, 8, 10, 12, 14])
            # egészséges lágyszövet
            healthy_soft_tissue_df = self.multi_filter(four_img_array[i][5], four_img_array[i][3], four_img_array[i][4],
                                                       lung_state="healthy_soft_tissue")
            # Töröljük a teljesen üres sorokat.
            healthy_soft_tissue_df = self.remove_null_rows(healthy_soft_tissue_df)
            # Kiválasztjuk azokat a sorokat, ahol a 'Label' oszlop értéke 1
            healthy_soft_tissue_df = self.select_random_rows(healthy_soft_tissue_df, [0, 1])

            if slice_counter == 1:
                df_all = diseased_lungs_df.copy()
                df_all = self.dataframe_merge_rows(df_all, healthy_lungs_df)
                df_all = self.dataframe_merge_rows(df_all, diseased_soft_tissue_df)
                df_all = self.dataframe_merge_rows(df_all, healthy_soft_tissue_df)
            else:
                df_all = self.dataframe_merge_rows(df_all, diseased_lungs_df)
                df_all = self.dataframe_merge_rows(df_all, healthy_lungs_df)
                df_all = self.dataframe_merge_rows(df_all, diseased_soft_tissue_df)
                df_all = self.dataframe_merge_rows(df_all, healthy_soft_tissue_df)
            slice_counter += 1

        # Ha a kép értéke nulla (fekete pixel), a címke értéke nulla.
        df_all.loc[df_all['Image'] == 0.0, 'Label'] = 0
        return df_all