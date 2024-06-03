from modules.utils import get_parameters, get_parameter


def main():
    print("""
    Insert option to continue:
          1. Split datasets
          2. Extract labels from dataset
          3. Test model
          4. Generate squared images from dataset
          5. Test detect circles
          6. (Training) Find Hough Parameters
          7. Run coin detection on one image
          8. Run coin recognition on one image
          9. Test coin detection on testing set.
          10. Test coin recognition on testing set.
          """)

    option = input("Option: ")

    match option:
        case "1":
            from modules.dataset.dataset_manipulation import split_and_persist_datasets
            split_and_persist_datasets(get_parameter("image_path"))
        case "2":
            from modules.dataset.dataset_manipulation import get_labels
            print(get_labels())
        case "3":
            from modules.model.model_tester import test_model
            test_model()
        case "4":
            from modules.dataset.dataset_manipulation import generate_coin_squares
            generate_coin_squares()
        case "5":
            from modules.model.circle_detection.hough_transform import detect_circles
            # detect_circles("../Images/28.jpg")
            from modules.model.model_tester import test_find_coins
            test_find_coins()
        case "6":
            from modules.model.model_training import find_hough_parameters 
            print(find_hough_parameters())
        case "7":
            from modules.model.circle_detection.hough_transform import detect_circles,extract_color_features, extract_color_and_hog_features, create_features_vector
            image_path = input("Insert image path: ")
            circles = detect_circles(image_path)  
            color_features = extract_color_features(image_path, circles) 
            print("Color features:", color_features)
        case "8":
            from modules.model.model_wrapper import detect_coins
            image_path = input("Insert image path: ")
            print(detect_coins(image_path, get_parameters()))
        case "9":
            from modules.model.model_tester import test_find_coins
            test_find_coins(dataset="testing_set")
        case "10":
            from modules.model.model_tester import test_model
            test_model(dataset="testing_set")
        case _:
            print("Invalid option")
            main()


main()
