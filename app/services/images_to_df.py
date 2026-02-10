import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from tqdm import tqdm
from skimage.filters import sobel


class ImagesToDf(object):
    def __init__(self):
        self.gabor_kernels = self.create_gabor_kernels()
        print(f"✅ Gabor kernelek száma: {len(self.gabor_kernels)}")

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
        Gabor-szűrő magok (kernels) generálása különböző paraméterekkel.

        Returns:
            list: OpenCV Gabor kernel objektumok listája.
        """
        kernels = []
        ksize = 3
        thetas = [0, np.pi / 4]
        sigmas = [1, 3]
        lamdas = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        gammas = [0.05, 0.5]
        psi = 0
        for theta in thetas:
            for sigma in sigmas:
                for lamda in lamdas:
                    for gamma in gammas:
                        kernel = cv2.getGaborKernel(
                            (ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F
                        )
                        kernels.append(kernel)
        return kernels

    @staticmethod
    def remove_null_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Eltávolítja a háttérnek minősülő (0 értékű) pixeleket a táblázatból.

        Args:
            df (pd.DataFrame): A bemeneti jellemzőtábla.

        Returns:
            pd.DataFrame: A megtisztított táblázat, amely csak releváns pixeleket tartalmaz.
        """
        if df.empty:
            return df

        # 1. Stratégia: Ha az EREDETI pixel (Image oszlop) 0, akkor az háttér.
        if 'Image' in df.columns:
            # Csak azokat tartjuk meg, ahol az eredeti kép abszolút értéke nagyobb mint 0
            # (1e-6 a lebegőpontos hibák miatt biztonságosabb mint a sima 0)
            df_cleaned = df[df['Image'].abs() > 1e-6]
            return df_cleaned

        # 2. Stratégia (Fallback): Ha valamiért nincs Image oszlop
        middle_cols = df.iloc[:, 1:-2]
        condition = ((middle_cols == 0) | (middle_cols.isna())).all(axis=1)
        df_cleaned = df[~condition]
        return df_cleaned

    @staticmethod
    def select_random_rows(df: pd.DataFrame, selected_values: list, n_limit: int = 2000) -> pd.DataFrame:
        """
        Véletlenszerű mintavételezés (downsampling) a kiegyensúlyozott adathalmazért.

        Címkénként korlátozza a pixelek számát, hogy a modell ne tanuljon el részrehajlást
        a túlreprezentált osztályok irányába.

        Args:
            df (pd.DataFrame): Forrás adatok.
            selected_values (list): A megtartandó Label értékek listája.
            n_limit (int): Maximális mintaszám címkénként. Alapértelmezett: 2000.

        Returns:
            pd.DataFrame: A mintavételezett adathalmaz.
        """
        if df.empty:
            return df

        filtered_dfs = []
        for value in selected_values:
            temp_df = df[df['Label'] == value]
            if not temp_df.empty:
                # Ha kevesebb pixel van, mint a limit, az összeset kéri,
                # ha több, akkor pontosan n_limit darabot.
                count = min(n_limit, len(temp_df))
                sampled_df = temp_df.sample(n=count, random_state=42)
                filtered_dfs.append(sampled_df)

        if not filtered_dfs:
            return pd.DataFrame(columns=df.columns)

        return pd.concat(filtered_dfs).sort_index()

    def multi_filter(self, patient_id, img, tumor_type, lung_state):
        """
               Különböző digitális képszűrők alkalmazása egy adott képre.

               Létrehozza a jellemzővektorokat: Gabor-válaszok, Sobel-élkeresés,
               Gauss-simítás, medián szűrő és variancia. Hozzárendeli a megfelelő
               címkét (Label) a tüdő állapota és a daganat típusa alapján.

               Args:
                   patient_id (str): A páciens azonosítója.
                   img (np.ndarray): A bemeneti kép (pixel array).
                   tumor_type (str): A daganat típusa (A, B, G, D).
                   lung_state (str): A szövet típusa (pl. 'diseased_lungs', 'healthy_soft_tissue').

               Returns:
                   pd.DataFrame: Egy táblázat, ahol minden sor egy pixel, az oszlopok pedig a szűrt értékek.
               """
        # Másolat készítése
        img2 = img.copy()

        df = pd.DataFrame()

        # 1. Oszlop: Eredeti pixel értékek
        df["Image"] = img2.reshape(-1)

        # 2. Gabor szűrők
        num = 1
        for gabor in self.gabor_kernels:
            gabor_label = 'Gabor' + str(num)
            fimg = cv2.filter2D(img2.astype('float32'), cv2.CV_32F, gabor)
            df[gabor_label] = fimg.reshape(-1)
            num += 1

        # 3. Egyéb szűrők
        df['Sobel'] = sobel(img2).reshape(-1)
        df['Gaussian_s3'] = nd.gaussian_filter(img2, sigma=3).reshape(-1)
        df['Gaussian_s7'] = nd.gaussian_filter(img2, sigma=7).reshape(-1)
        df['Median_s3'] = nd.median_filter(img2, size=3).reshape(-1)
        df['Variance_s3'] = nd.generic_filter(img2, np.var, size=3).reshape(-1)

        # --- Címkézés ---
        label_value = 0
        if lung_state == "healthy_lungs":
            label_value = 1
        elif lung_state == "diseased_lungs":
            if tumor_type == 'A':
                label_value = 4
            elif tumor_type == 'B':
                label_value = 5
            elif tumor_type == 'D':
                label_value = 6
            elif tumor_type == 'G':
                label_value = 7
        elif lung_state == "healthy_soft_tissue":
            label_value = 1
        elif lung_state == "diseased_soft_tissue":
            if tumor_type == 'A':
                label_value = 8
            elif tumor_type == 'B':
                label_value = 10
            elif tumor_type == 'D':
                label_value = 12
            elif tumor_type == 'G':
                label_value = 14

        df["Label"] = label_value
        # JAVÍTÁS: Biztosítjuk, hogy a patient_id minden sorba bekerüljön
        df["patient_id"] = str(patient_id)

        return df

    def preproccessing_images(self, four_img_array):
        """
            Optimalizált képfeldolgozó, jellemzőkinyerő és statisztikakészítő függvény.

            Bemenet:
            four_img_array: listák listája (CT szeletek és metaadatok)
            Kimenet:
            pd.DataFrame: Az összesített, szűrt és címkézett adathalmaz.
            """
        dfs_to_merge = []

        for i in tqdm(range(len(four_img_array)), desc="Preproccessing images"):
            try:
                # Változók kibontása
                img_orig = four_img_array[i][0]
                img_par = four_img_array[i][1]
                img_tum = four_img_array[i][2]
                img_roi = four_img_array[i][3]
                label = four_img_array[i][4]
                p_id = four_img_array[i][5]

                # 1. Beteg tüdő (eredeti kép)
                df_diseased = self.multi_filter(p_id, img_orig, label, lung_state="diseased_lungs")
                df_diseased = self.remove_null_rows(df_diseased)
                df_diseased = self.select_random_rows(df_diseased, [0, 4, 5, 6, 7])

                # 2. Egészséges tüdő (szegmentált parenchyma)
                df_healthy_lung = self.multi_filter(p_id, img_par, label, lung_state="healthy_lungs")
                df_healthy_lung = self.remove_null_rows(df_healthy_lung)
                df_healthy_lung = self.select_random_rows(df_healthy_lung, [0, 1])

                # 3. Beteg lágyszövet (szegmentált tumor)
                df_diseased_soft = self.multi_filter(p_id, img_tum, label, lung_state="diseased_soft_tissue")
                df_diseased_soft = self.remove_null_rows(df_diseased_soft)
                df_diseased_soft = self.select_random_rows(df_diseased_soft, [0, 8, 10, 12, 14])

                # 4. Egészséges lágyszövet (tumor körüli terület)
                df_healthy_soft = self.multi_filter(p_id, img_roi, label, lung_state="healthy_soft_tissue")
                df_healthy_soft = self.remove_null_rows(df_healthy_soft)
                df_healthy_soft = self.select_random_rows(df_healthy_soft, [0, 1])

                # Rész-adatkeretek hozzáadása a listához
                dfs_to_merge.extend([df_diseased, df_healthy_lung, df_diseased_soft, df_healthy_soft])

            except Exception as e:
                print(f"⚠️ Hiba a(z) {i}. indexű szeletnél (Páciens: {four_img_array[i][5]}): {e}")

        if not dfs_to_merge:
            return pd.DataFrame()

        # --- ÖSSZESÍTÉS ÉS UTÓMUNKA ---
        df_all = pd.concat(dfs_to_merge, ignore_index=True)

        # Ha a pixel értéke 0, a Label is legyen 0
        if 'Image' in df_all.columns:
            df_all.loc[df_all['Image'] == 0.0, 'Label'] = 0

        # --- PÁCIENS SZINTŰ STATISZTIKA ---
        # Ez a rész segít ellenőrizni az adatok egyensúlyát (class balance)
        print("\n" + "=" * 60)
        print(f"{'📊 PÁCIENS SZINTŰ STATISZTIKA':^60}")
        print("=" * 60)

        # Egyedi páciens azonosítók kigyűjtése
        patient_stats = df_all['patient_id'].value_counts()

        for p_name, total_count in patient_stats.items():
            # Csak az adott páciens sorait nézzük
            p_mask = (df_all['patient_id'] == p_name)

            # Tumoros pixelek száma (ahol a Label > 1)
            tumor_pixel_count = len(df_all[p_mask & (df_all['Label'] > 1)])

            # Tumoros pixelek aránya (LaTeX formátumban: $Ratio = \frac{Tumor}{Total}$)
            ratio = (tumor_pixel_count / total_count) * 100 if total_count > 0 else 0

            print(
                f"👤 Páciens: {p_name:<20} | Összes: {total_count:>6} | Tumor: {tumor_pixel_count:>6} ({ratio:>5.1f}%)")

        print("-" * 60)
        print(f"📈 ÖSSZESEN: {len(df_all)} pixel került feldolgozásra.")
        print("=" * 60 + "\n")

        return df_all
