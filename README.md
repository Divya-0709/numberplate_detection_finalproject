# numberplate_detection_finalproject
Overview
This project implements an end-to-end Automatic License Plate Recognition (ALPR) system using deep learning and computer vision techniques. The system detects vehicle number plates from images and extracts alphanumeric text, enabling real-time and scalable vehicle identification.

Techniques Used
1. Object Detection (YOLOv8)
•	YOLOv8 is used for real-time number plate detection.
•	A pretrained YOLOv8 model is fine-tuned on a custom dataset annotated in YOLO format.
•	Single-class detection (plate) improves localization accuracy and reduces model complexity.
•	YOLO’s one-stage architecture ensures fast inference suitable for real-world deployment.
________________________________________
2. Data Preparation & Handling
•	Images and annotations are validated to ensure one-to-one mapping.
•	Dataset is split into 80% training, 10% validation, and 10% testing.
•	Unmatched images and labels are identified and removed to maintain dataset integrity.
•	Bounding box statistics are analyzed to understand object scale and positioning.
________________________________________
3. Model Training & Evaluation
•	The model is trained using GPU acceleration for efficient convergence.
•	Performance is evaluated using:
o	Precision, Recall
o	F1-score
o	mAP@50 and mAP@50–95
o	Confusion Matrix
•	Training and validation losses are monitored to prevent overfitting.
________________________________________
4. Image Preprocessing (OpenCV)
•	Detected plate regions are preprocessed before OCR using:
o	Grayscale conversion
o	Image upscaling
o	Adaptive/Otsu thresholding
•	These steps enhance character clarity and improve OCR accuracy.
________________________________________
5. Optical Character Recognition (EasyOCR)
•	EasyOCR, a deep learning–based OCR engine, is used for text extraction.
•	It handles skewed, low-resolution, and real-world license plate images effectively.
•	OCR results are combined with detection confidence for reliable output.
________________________________________
6. Deployment (Streamlit)
•	A Streamlit web application provides a user-friendly interface.
•	Features include:
o	Image upload and camera input
o	Real-time detection
o	Annotated image display
o	OCR text output
o	Downloadable results (CSV and images)

•	Conclusion
The project successfully demonstrates a robust and scalable ALPR solution by combining YOLOv8 for accurate number plate detection and EasyOCR for reliable text recognition. The system achieves high detection accuracy, strong precision, and real-time performance, making it suitable for applications such as traffic monitoring, parking systems, toll collection, and security surveillance. With further dataset expansion and optimization, the solution can be easily adapted for large-scale and region-specific deployments.

