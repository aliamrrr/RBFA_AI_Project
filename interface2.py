def show_interface_2(): 
        import streamlit as st
        import tempfile
        from utils.models import load_player_detection_model, load_field_detection_model
        from utils.team_classifier import show_crops, extract_player_crops, fit_team_classifier
        from PIL import Image
        from utils.frame_process import generated_lines_video, process_frame, draw_team_connections, process_frame_with_convex_hull, draw_radar_view,extract_tracking_data,annotate_frame_with_hulls,annotate_player_name,passes_options,calculate_optimal_passes,calculate_realistic_optimal_passes
        import supervision as sv
        import cv2
        import os
        from utils.soccer import SoccerPitchConfiguration
        import os
        import svgwrite as svg

        def hex_to_svg_color(hex_color):
            hex_color = hex_color.lstrip('#')
            return sv.Color.from_hex(hex_color.upper())


        CONFIG = SoccerPitchConfiguration()
        tracker = sv.ByteTrack()
        tracker.reset()

        # Sidebar Menu
        menu = st.sidebar.selectbox("Select an option", 
                                    ["Football Vision","Other"])

        # 1. **Football Vision Tab**
        if menu == "Football Vision":
            tracking_option = st.selectbox("Choose a feature", 
                                        ["View the video", "Collect, classify and view teams", 
                                        "Football Video Visualizations","Tactical visualizations"])

            # 1.1 **View the video**
            if tracking_option == "View the video":
                uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4","mkv"])

                if uploaded_file is not None:
                    # Creates a temporary file for the uploaded video
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    # Reads and display the video
                    st.video(tmp_file_path)


                # 1.2 **Collect, classify and view teams**
            elif tracking_option == "Collect, classify and view teams":
                    uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4","mkv"])

                    if uploaded_file is not None:
                        # Creates a temporary file for the uploaded video
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name

                        # Loads player and field detection models
                        direct_link = "https://drive.google.com/file/d/1FuibHhLGI7PvaZxSPrxhtdQxveyqdKTg"

                        player_detection_model = load_player_detection_model(direct_link=direct_link)  # Load the player model
                        field_detection_model = load_field_detection_model()    # Load the field model
                        print(player_detection_model)
                        # Displays a button to start player detection
                        if st.button("Detect players"):
                            # Shows a loading indicator during detection
                            with st.spinner("Detection in progress..."):
                                # Extracts player crops
                                crops = extract_player_crops(tmp_file_path, player_detection_model,max_frames=10)
                                
                                # Checks if crops were detected
                                if crops:
                                    # Ensures the 'crops' directory exists
                                    crops_directory = "crops"
                                    os.makedirs(crops_directory, exist_ok=True)

                                    # Saves each crop as an image in the 'crops' directory
                                    for i, crop in enumerate(crops):
                                        crop_path = os.path.join(crops_directory, f"crop_{i+1}.jpg")

                                        pil_crop = Image.fromarray(crop)

                                        pil_crop.save(crop_path)

                                    show_crops(crops)
                                    st.write(f"Number of players detected: {len(crops)}")
                                    with st.spinner("Classification in progress..."):
                                        try:
                                            # Prepare team classifier
                                            team_classifier = fit_team_classifier(crops, device="cpu")[0]
                                            st.session_state.team_classifier = team_classifier
                                            st.success("Team classification model is ready!")
                                            st.write("You can now view the teams.")
                                        except Exception as e:
                                            st.error(f"Error during classification: {str(e)}")
                                else:
                                    st.warning("No players detected. Please try with another video.")

            elif tracking_option == "Football Video Visualizations":
                        orig_frame = None
                        type = 'tactical'
                        players_tracking, ball_tracking = None, None
                        
                        input_type = st.radio("Select input type", ["Video", "Image"])
                        
                        if input_type == "Video":
                            uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4", "mkv"])
                        else:
                            uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
                    
                        if uploaded_file is not None:
                            if input_type == "Video":
                                suffix = ".mp4"
                            else:
                                suffix = "." + uploaded_file.name.split('.')[-1]

                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_file_path = tmp_file.name

                            if "team_classifier" not in st.session_state:
                                st.warning("Please first classify the players in the previous tab.")
                            else:
                                team_classifier = st.session_state.team_classifier

                            # two options as an input (Image or video)
                            if input_type == "Video":
                                    cap = cv2.VideoCapture(tmp_file_path)
                                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    cap.release()

                                    total_duration_seconds = total_frames / fps

                                    time_seconds = st.slider("Select the time", 0, int(total_duration_seconds), 0)

                                    minutes = time_seconds // 60
                                    seconds = time_seconds % 60

                                    st.write(f"Time: {minutes} minutes {seconds} seconds")

                                    frame_index = int(time_seconds * fps)

                                    annotated_frame = None

                                    cap = cv2.VideoCapture(tmp_file_path)
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                                    ret, frame = cap.read()
                                    orig_frame = frame

                            else:
                                    annotated_frame = None
                                    frame = cv2.imread(tmp_file_path)
                                    ret = True
                                    orig_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            team_id = st.selectbox("Select team (0 or 1) to view convex hull", [0, 1])
                            use_convex_area_on_pitch = st.checkbox("Use convex hull area on pitch", value=True)
                            use_convex_area_on_frame = st.checkbox("Use convex hull area on frame", value=False)
                            save_convex_area_video = st.checkbox("Save convex hull area video", value=False)

                            selected_color_1 = st.color_picker("Select team 0 Color", "#FFFF00")
                            selected_color_2 = st.color_picker("Select team 1 Color", "#FF0000")

                            color_1 = hex_to_svg_color(selected_color_1)
                            color_2 = hex_to_svg_color(selected_color_2)

                            if ret:
                                    st.image(orig_frame, channels="BGR", caption=f"Frame")

                                    if use_convex_area_on_pitch:
                                        annotated_frame = annotate_frame_with_hulls(orig_frame, team_id, team_classifier, CONFIG,color_1,color_2)
                                    elif use_convex_area_on_frame:
                                        annotated_frame = process_frame_with_convex_hull(orig_frame, team_classifier, team_id=None)
                                    elif save_convex_area_video:
                                        generated_lines_video(tmp_file_path, team_classifier,CONFIG,fps_output=30)
                                    else:
                                        annotated_frame = process_frame(orig_frame, team_classifier,color_1,color_2)

                                    st.image(annotated_frame, channels="BGR", caption=f"Annotated frame")
                            else:
                                    st.error(f"Error: Unable to extract frame {frame_index}. Try selecting another time.")

            # 1.3 **View an annotated frame**
            elif tracking_option == "Tactical visualizations":
                orig_frame = None
                type = 'tactical'
                players_tracking, ball_tracking = None, None
                
                # Video or Image
                input_type = st.radio("Select input type", ["Video", "Image"])
                
                if input_type == "Video":
                    uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4", "mkv"])
                else:
                    uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
            
                if uploaded_file is not None:
                    if input_type == "Video":
                        suffix = ".mp4"
                    else:
                        suffix = "." + uploaded_file.name.split('.')[-1]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    if "team_classifier" not in st.session_state:
                        st.warning("Please first classify the players in the previous tab.")
                    else:
                        team_classifier = st.session_state.team_classifier

                        if input_type == "Video":
                            cap = cv2.VideoCapture(tmp_file_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()

                            total_duration_seconds = total_frames / fps

                            time_seconds = st.slider("Select the time", 0, int(total_duration_seconds), 0)

                            minutes = time_seconds // 60
                            seconds = time_seconds % 60

                            st.write(f"Time: {minutes} minutes {seconds} seconds")

                            frame_index = int(time_seconds * fps)

                            annotated_frame = None

                            cap = cv2.VideoCapture(tmp_file_path)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                            ret, frame = cap.read()
                        else:
                            annotated_frame = None
                            frame = cv2.imread(tmp_file_path)
                            ret = True
                            orig_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        if ret and orig_frame is not None:
                            st.image(orig_frame, channels="BGR" if input_type == "Video" else "RGB", caption=f"Frame {frame_index}" if input_type == "Video" else "Uploaded Image")

                        if 'passes_mode' not in st.session_state:
                            st.session_state.passes_mode = None
                        
                        team_id = 0
                        team_id = st.selectbox("Select team (0 or 1) to view connections", [0, 1])

                        selected_color_1 = st.color_picker("Select team 0 Color", "#FFFF00")
                        selected_color_2 = st.color_picker("Select team 1 Color", "#FF0000")

                        color_1 = hex_to_svg_color(selected_color_1)
                        color_2 = hex_to_svg_color(selected_color_2)

                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            if st.button("Generate team connections"):
                                annotated_frame = draw_team_connections(frame, CONFIG, team_classifier, team_id,color_1,color_2)

                        with col2:
                            if st.button("Generate Voronoi diagram"):
                                type = 'voronoi'
                                print(type)
                                annotated_frame = draw_radar_view(frame, CONFIG, team_classifier, type,color_1,color_2)

                        with col4:
                            if st.button("Pass options : Play short"):
                                passes_mode = 'build'
                                mode = 1
                                print(CONFIG)
                                annotated_frame = passes_options(frame, CONFIG, team_classifier, passes_mode,color_1,color_2)
                        mode = 1
                        if annotated_frame is not None and mode == 1:
                            st.image(annotated_frame, channels="BGR", caption=f"Annotated frame")

                        with col5:
                            if st.button("Pass option : Safe and high"):
                                passes_mode = 'interception'
                                mode = 2
                                annotated_frame = passes_options(frame, CONFIG, team_classifier, passes_mode,color_1,color_2)
                        
                        if annotated_frame is not None and mode == 2:
                            st.image(annotated_frame, channels="BGR", caption=f"Annotated frame")

                        with col3:
                         if st.button("Preview Tracking dataframes"):
                                    players_tracking, ball_tracking = extract_tracking_data(frame, CONFIG, team_classifier, tracker)
                                
                                    players_tracking['player_x'] /= 100
                                    players_tracking['player_y'] /= 100

                                    ball_tracking['ball_x'] /= 100
                                    ball_tracking['ball_y'] /= 100
                        
                         if players_tracking is not None and ball_tracking is not None:
                            column1, column2 = st.columns(2)
                                    
                            with column1:
                                st.subheader("Players Tracking Data")
                                st.dataframe(players_tracking)

                            with column2:
                                st.subheader("Ball Tracking Data")
                                st.dataframe(ball_tracking)
                            
                         if input_type == "Video":
                            cap.release()


