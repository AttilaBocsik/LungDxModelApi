import os
import numpy as np
import pydicom
import SimpleITK as sitk
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from .XML_preprocessor import XML_preprocessor
from skimage.draw import rectangle_perimeter
from skimage import measure, segmentation
from skimage.segmentation import active_contour
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters import gaussian
from scipy import ndimage
from .images_to_df import ImagesToDf
from app.core.config import settings  # Importáld a beállításokat

class DicomManager(object):
    def __init__(self):
        class_list = self.get_category(settings.CATEGORY_FILE)
        self.num_classes = len(class_list)

    def getUID_path(self, dicom_path):
        """
        DICOM fájlok sorbarendezése és metaadatok kinyerése
        :param path: DICOM fájlok könyvtára
        :return: DICOM fájlok metaadatai files information to python dict
        """
        dict = {}
        info = self.load_file_information(dicom_path)
        dict[info['dicom_num']] = dicom_path
        return dict

    @staticmethod
    def get_category(category_file):
        """
        Kategóriák beolvasása hibatűréssel
        """
        if not os.path.exists(category_file):
            raise FileNotFoundError(f"Kritikus hiba: A kategória fájl nem található itt: {category_file}")

        class_list = []
        with open(category_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_list.append(line.strip())
        return class_list

    @staticmethod
    def show_image(img, title='image', t=0, esc=False):
        """
        Show image
        :param img: OpenCV image
        :param title: Image title
        :param t: Timer
        :param esc: ESC button activate
        :return: Show image
        """
        cv2.imshow(title, img)
        if esc:
            while cv2.waitKey(0) != 27:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) <= 0:
                    break
        else:
            cv2.waitKey(t)
        cv2.destroyWindow(title)

    @staticmethod
    def class_colors(num_colors):
        """
        Generate color from label list
        :param num_colors: Label list
        :return: Class color
        """
        class_colors = []
        for i in range(0, num_colors):
            hue = 255 * i / num_colors
            col = np.zeros((1, 1, 3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128  # Saturation
            col[0][0][2] = 255  # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            class_colors.append(col)

        return class_colors

    def roi2rect(self, img_name, img_np, img_data, label_list, image):
        """
        It prepares the mask and position of the tumor determined by the doctors.
        :param img_name: Image name
        :param img_np: Numpy array to the input image
        :param img_data: ROI position
        :param label_list: Tumor type list
        :param image: Image
        :return: ROI segmented image, ROI rectangle position
        """
        colors = self.class_colors(len(label_list))
        for rect in img_data:
            # Az orvosok alltal behatarolt daganat
            bounding_box = [rect[0], rect[1], rect[2], rect[3]]
            xmin = int(bounding_box[0])
            ymin = int(bounding_box[1])
            xmax = int(bounding_box[2])
            ymax = int(bounding_box[3])
            pmin = (xmin, ymin)
            pmax = (xmax, ymax)
            width = xmax - xmin
            height = ymax - ymin
            rectangle_position = {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "width": width,
                "height": height,
            }

            color = (255, 255, 255)
            # Initialize the mask
            roi = np.zeros(image.shape[:2], np.uint8)
            roi = cv2.rectangle(roi, pmin, pmax, color, cv2.FILLED)
            mask = np.ones_like(image) * 255
            mask = cv2.bitwise_and(mask, image, mask=roi) + cv2.bitwise_and(mask, mask, mask=~roi)
            label_array = rect[4:]
            index = int(np.where(label_array == float(1))[0])
            label = label_list[index]
            '''
            if label == 'A':
                label = 'Adenocarcinoma'
            elif label == 'B':
                label = 'Small Cell Carcinoma'
            elif label == 'D':
                label = 'Large Cell Carcinoma'
            elif label == 'G':
                label = 'Squamous Cell Carcinoma'
            '''
            # color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))
            color = colors[index]
            cv2.rectangle(img_np, pmin, pmax, color, 1)
        # showImage(img=img_np, title=img_name)
        # Apply thresholding (adjust threshold value as needed)
        _, segmented_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return segmented_image, rectangle_position, label

    @staticmethod
    def matrix_to_image(data, ch):
        """
        If, DICOM image type CT
        :param data:
        :param ch:
        :return:
        """
        # Data = (Data+1024)*0.125
        # new_im = Image.fromarray(Data.astype(np.uint8))
        # new_im.show()
        if ch == 3:
            img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if ch == 1:
            data = (data + 1024) * 0.125
            img_rgb = data.astype(np.uint8)
        return img_rgb

    @staticmethod
    def load_file_information(filename):
        """
        Read DICOM file information
        :param filename:
        :return:
        """
        information = {}
        ds = pydicom.dcmread(filename, force=True)

        information['dicom_num'] = ds.SOPInstanceUID
        information['PatientPosition'] = ds.PatientPosition
        information['PatientID'] = ds.PatientID
        information['PatientName'] = ds.PatientName
        # information['PatientBirthDate'] = ds.PatientBirthDate
        # information['PatientSex'] = ds.PatientSex
        information['StudyID'] = ds.StudyID
        # information['StudyDate'] = ds.StudyDate
        # information['StudyTime'] = ds.StudyTime
        # information['InstitutionName'] = ds.InstitutionName
        # information['Manufacturer'] = ds.Manufacturer
        # information['NumberOfFrames'] = ds.NumberOfFrames
        return information

    @staticmethod
    def load_file(filename):
        """
        Feldolgozza a DICOM fájlt
        :param filename: DICOM fájl elérési utja
        :return: ds, img, Numpy tömb(img_array), frame_num, width, height, ch
        """
        ds = pydicom.dcmread(filename)
        img = sitk.ReadImage(filename)
        # Konvertaljuk a Dicom kepet NumPy tombbe
        img_array = sitk.GetArrayFromImage(img)
        if len(img_array.shape) == 3:
            frame_num, width, height = img_array.shape
            ch = 1
            return ds, img, img_array, frame_num, width, height, ch
        elif len(img_array.shape) == 4:
            frame_num, width, height, ch = img_array.shape
            return ds, img, img_array, frame_num, width, height, ch

    @staticmethod
    def gvf_snake(image, rectangle_position):
        """
        Ez az algoritmus határozza meg a daganat ROI-ját a területen.
        :param image: Szürkeárnyalatos opencv kép numpy tömb alakja.
        :param rectangle_position:  Regions of Interest(ROI) pozicó
        :return: Illustrative image, Tumor points, ROI points
        """
        # image = ds.pixel_array.astype('float32')
        # Let's normalize the image between 0 and 1
        image -= np.min(image)
        image /= np.max(image)

        # Initialize the contour
        '''
        s = np.linspace(0, 2 * np.pi, 400)
        r = rows / 2 + rows / 4 * np.sin(s)
        c = columns / 2 + columns / 4 * np.cos(s)
        init = np.array([r, c]).T
        '''
        # Define the coordinates of the rectangle
        rr, cc = rectangle_perimeter((rectangle_position["ymin"] - 3, rectangle_position["xmin"] - 3),
                                     end=(rectangle_position["ymax"] + 3, rectangle_position["xmax"] + 3),
                                     shape=image.shape)  # (row, column)
        # Initialize the snake with the rectangle coordinates
        init = np.array([rr, cc]).T
        # We execute the GVF Snake algorithm
        # snake = active_contour(image, init, alpha=0.015, beta=10, gamma=0.001), preserve_range=False
        snake = active_contour(gaussian(image, 3), init, alpha=0.01, beta=3, gamma=0.001)
        # We draw the final contour
        final_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        init_array = []
        for point in init.astype(int):
            cv2.circle(final_image, (point[1], point[0]), 1, (255, 6, 0), -1)
            init_array.append([point[1], point[0]])
        init_points = np.array(init_array)
        snake_array = []
        for point in snake.astype(int):
            cv2.circle(final_image, (point[1], point[0]), 1, (0, 255, 0), -1)
            snake_array.append([point[1], point[0]])
        snake_points = np.array(snake_array)
        return final_image, snake_points, init_points

    @staticmethod
    def get_pixels_hu(scans):
        """
        Converts raw images to Hounsfield Units (HU).
        Parameters: scans (Raw images)
        Returns: image (NumPy array)
        """
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)
        # Since the scanning equipment is cylindrical in nature and image output is square,
        # we set the out-of-scan pixels to 0
        image[image == -2000] = 0
        # HU = m*P + b
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    @staticmethod
    def load_scans(paths):
        """
        Loads scans from a folder and into a list.
        Parameters: path (Folder path)
        Returns: slices (List of slices)
        """
        slices = []
        for path in paths:
            slices.append(pydicom.dcmread(path, force=True))
        slices.sort(key=lambda x: int(x.InstanceNumber))
        try:
            if len(paths) == 1:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2])
            else:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            if len(paths) == 1:
                slice_thickness = np.abs(slices[0].SliceLocation)
            else:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    @staticmethod
    def generate_markers(image, hu):
        """
        Jelölőket generál egy adott képhez.
        Parameters: image
        Returns: Internal Marker, External Marker, Watershed Marker
        """
        # Creation of the internal Marker
        marker_internal = image < hu
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0
        # Creation of the External Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        # Creation of the Watershed Marker
        marker_watershed = np.zeros((512, 512), dtype=np.int32)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128
        return marker_internal, marker_external, marker_watershed

    def make_lungmask(self, slice_dcm_path_list, hu=-400):
        """
        Szeletfájl-maszk létrehozása
        :param slice_dcm_path_list:
        :param hu:
        :return: Szelet fájlok maszk tömb
        """
        train_patient_scans = self.load_scans(slice_dcm_path_list)
        train_patient_images = self.get_pixels_hu(train_patient_scans)
        test_patient_internal_list = []
        for imgi in range(len(train_patient_images[:])):
            test_patient_internal, test_patient_external, test_patient_watershed = self.generate_markers(
                train_patient_images[imgi], hu)
            test_patient_internal_list.append(test_patient_internal)
        return test_patient_internal_list

    def preprocessing_dicom(self, dicom_path, annotation_path) -> pd.DataFrame | None:
        dicom_path = Path(rf"{dicom_path}")
        dicom_filename = os.path.basename(dicom_path)
        information = self.load_file_information(dicom_path)
        patient_position = information['PatientPosition']
        patient_id = information['PatientID']
        if patient_position != 'HFS':
            return None

        ds, img, matrix, frame_num, width, height, ch = self.load_file(dicom_path)  # load DICOM file
        img_bitmap = self.matrix_to_image(matrix[0], ch)
        origin_img = ds.pixel_array.astype('float32')
        patient_slice = self.getUID_path(dicom_path)  # Type: <class 'pydicom.uid.UID'>
        annotations = XML_preprocessor(annotation_path, num_classes=self.num_classes).data
        k, v = list(annotations.items())[0]
        print(f"Kulcs: {k}, Érték: {v}, DICOM név: {dicom_filename}")
        try:
            for x, y in patient_slice.items():
                print(x, y)
        except KeyError as e:
            msg = f"Error: {e}"
            print(msg)

        tumor_mask_ndarray, roi_rectangle_position, tumor_mask_label = self.roi2rect(
            img_name=dicom_filename,
            img_np=img_bitmap,
            img_data=v,
            label_list=['A', 'B', 'D', 'G'],
            image=origin_img)
        # tumor_mask_label tartalma => A vagy B vagy D vagy G
        tumor_mask_img = cv2.cvtColor(tumor_mask_ndarray, cv2.COLOR_RGB2BGR)
        tumor_mask_img_gray = cv2.cvtColor(tumor_mask_img, cv2.COLOR_BGR2GRAY)
        # Segmentation tumor at lung with snake contour
        tumor_img, snake_points, roi_points = self.gvf_snake(tumor_mask_img_gray, roi_rectangle_position)
        # Készítsen maszkot a kígyó végső kontúrjából
        tumor_mask = np.zeros_like(tumor_mask_img_gray)
        cv2.fillPoly(tumor_mask, pts=[snake_points], color=(255, 0, 0))
        masked_tumor = tumor_mask * origin_img
        # Hozzon létre egy maszkot a végső ROI-kontúrból
        roi_mask = np.zeros_like(tumor_mask_img_gray)
        cv2.fillPoly(roi_mask, pts=[roi_points], color=(255, 0, 0))
        masked_roi = roi_mask * origin_img
        # A ROI-ban szegmentált beteg lágyszövet inverzéz is kiemeljük
        inverted_mask = np.ones_like(masked_roi) * 255
        cv2.fillPoly(inverted_mask, pts=[snake_points], color=(0, 0, 0))
        inverted_masked_roi = cv2.bitwise_and(masked_roi, inverted_mask)
        #  (Egészséges tüdő)
        mask_list_400 = self.make_lungmask([dicom_path], -400)
        segmented_parenchyma = mask_list_400[0] * origin_img
        # Vizuális rész
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        selected_origin_img = origin_img / np.max(origin_img)
        '''
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(15, 15))
        ax1.imshow(selected_origin_img, cmap='gray')
        ax1.set_title("Eredeti felvétel")
        ax1.axis('off')
        # Az "egészséges tüdő" címkét olyan részlet kapja, ahol a szegmentált maszk szerint tüdő van és nincs bejelölve.
        ax2.imshow(segmented_parenchyma, cmap='gray')  # cv2.medianBlur(segmented_parenchyma, 3)
        ax2.set_title("Szegmentált tüdő")
        ax2.axis('off')
        # Orvosi megjelölés
        ax3.imshow(masked_tumor, cmap='gray')
        ax3.set_title("Orvosi megjelölés")
        ax3.axis('off')
        ax4.imshow(inverted_masked_roi, cmap='gray')
        ax4.set_title("Orvosi megjelölés inverze")
        ax4.axis('off')
        plt.show()
        '''
        four_pictures_array = []
        four_pictures = [origin_img, segmented_parenchyma, masked_tumor, inverted_masked_roi, tumor_mask_label,
                         patient_id]
        four_pictures_array.append(four_pictures)
        itd = ImagesToDf()
        df = itd.preproccessing_images(four_pictures_array)
        print(f"Rows, Columns: {df.shape}")
        return df
