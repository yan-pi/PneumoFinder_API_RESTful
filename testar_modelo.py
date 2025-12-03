from service.pneumonia_service import PneumoniaDetectorService

if __name__ == "__main__":
    detector = PneumoniaDetectorService("models/best_model.keras")
    class_name, confidence, folder = detector.diagnose_with_explanation(
        "imgs/person75_bacteria_365.jpeg"
    )

    print(f"\nDiagnosis: {class_name} ({confidence:.1%})")
    print(f"Images saved in: {folder}")

    detector = PneumoniaDetectorService("models/best_model.keras")
    class_name, confidence, folder = detector.diagnose_with_explanation(
        "imgs/person72_bacteria_352.jpeg"
    )

    print(f"\nDiagnosis: {class_name} ({confidence:.1%})")
    print(f"Images saved in: {folder}")

    detector = PneumoniaDetectorService("models/best_model.keras")
    class_name, confidence, folder = detector.diagnose_with_explanation(
        "imgs/person74_bacteria_361.jpeg"
    )

    print(f"\nDiagnosis: {class_name} ({confidence:.1%})")
    print(f"Images saved in: {folder}")
