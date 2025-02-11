from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from utils.soccer import SoccerPitchConfiguration


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        background_color (sv.Color, optional): Color of the pitch background.
            Defaults to sv.Color(34, 139, 34).
        line_color (sv.Color, optional): Color of the pitch lines.
            Defaults to sv.Color.WHITE.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the pitch lines in pixels.
            Defaults to 4.
        point_radius (int, optional): Radius of the penalty spot points in pixels.
            Defaults to 8.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the soccer pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image

import cv2
import numpy as np
from typing import Optional

def draw_arrow_on_pitch(
    config: SoccerPitchConfiguration,
    xy_start: np.ndarray,
    xy_end: np.ndarray,
    color: sv.Color = sv.Color.RED,
    thickness: int = 2,
    scale: float = 0.1,
    padding: int = 50,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws arrows on a soccer pitch to indicate direction.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy_start (np.ndarray): Starting points of the arrows (x, y).
        xy_end (np.ndarray): Ending points of the arrows (x, y).
        color (sv.Color, optional): Color of the arrows. Defaults to sv.Color.RED.
        thickness (int, optional): Thickness of the arrow lines. Defaults to 2.
        scale (float, optional): Scaling factor for the pitch dimensions. Defaults to 0.1.
        padding (int, optional): Padding around the pitch in pixels. Defaults to 50.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw arrows on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with arrows drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for start, end in zip(xy_start, xy_end):
        scaled_start = (
            int(start[0] * scale) + padding,
            int(start[1] * scale) + padding
        )
        scaled_end = (
            int(end[0] * scale) + padding,
            int(end[1] * scale) + padding
        )
        
        # Draw the arrow line
        cv2.arrowedLine(
            img=pitch,
            pt1=scaled_start,
            pt2=scaled_end,
            color=color.as_bgr(),
            thickness=thickness,
            tipLength=0.1
        )

    return pitch


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with paths drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return pitch
    
    
def draw_pitch_hull(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,  # Tableau contenant les coordonnées des points
    team_ids: np.ndarray,  # Tableau contenant les IDs des équipes correspondant aux points
    team_id: Optional[int] = None,  # ID de l'équipe à tracer (None pour tracer toutes)
    color: sv.Color = sv.Color.RED,  # Couleur de remplissage de l'enveloppe
    edge_color: sv.Color = sv.Color.BLACK,  # Couleur des contours de l'enveloppe
    alpha: float = 0.5,  # Opacité de l'enveloppe
    thickness: int = 2,  # Épaisseur du contour
    padding: int = 50,  # Padding autour du terrain
    scale: float = 0.1,  # Facteur de mise à l'échelle
    pitch: Optional[np.ndarray] = None  # Image existante du terrain
) -> np.ndarray:
    """
    Dessine l'enveloppe convexe pour une ou plusieurs équipes sur un terrain de football.

    Args:
        config (SoccerPitchConfiguration): Configuration du terrain.
        xy (np.ndarray): Tableau des coordonnées des points (N, 2).
        team_ids (np.ndarray): Tableau des IDs des équipes associés à chaque point.
        team_id (Optional[int], optional): ID de l'équipe à tracer. None pour toutes les équipes.
            Defaults to None.
        color (sv.Color, optional): Couleur de remplissage des enveloppes.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Couleur des contours des enveloppes.
            Defaults to sv.Color.BLACK.
        alpha (float, optional): Opacité des enveloppes. Defaults to 0.5.
        thickness (int, optional): Épaisseur des contours. Defaults to 2.
        padding (int, optional): Padding autour du terrain en pixels. Defaults to 50.
        scale (float, optional): Facteur de mise à l'échelle. Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Image existante du terrain.
            Defaults to None.

    Returns:
        np.ndarray: Image du terrain avec les enveloppes convexes dessinées.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    def draw_hull(points_array: np.ndarray, color: sv.Color):
        if len(points_array) < 3:
            return  # Pas assez de points pour une enveloppe convexe
        scaled_points = [
            (int(p[0] * scale) + padding, int(p[1] * scale) + padding)
            for p in points_array
        ]
        hull = cv2.convexHull(np.array(scaled_points, dtype=np.int32))
        cv2.fillPoly(pitch, [hull], color=color.as_bgr())
        cv2.polylines(pitch, [hull], isClosed=True, color=edge_color.as_bgr(), thickness=thickness)

    if team_id is not None:
        # Filtrer les points de l'équipe spécifiée
        team_points = xy[team_ids == team_id]
        draw_hull(team_points, color)
    else:
        # Tracer les enveloppes convexes pour toutes les équipes
        unique_team_ids = np.unique(team_ids)
        for t_id in unique_team_ids:
            team_points = xy[team_ids == t_id]
            draw_hull(team_points, color)

    return pitch

def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams, and adds a text legend showing the proportion of the field controlled by each team.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 1.
        team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 2.
        team_1_color (sv.Color, optional): Color representing the control area of
            team 1. Defaults to sv.Color.RED.
        team_2_color (sv.Color, optional): Color representing the control area of
            team 2. Defaults to sv.Color.WHITE.
        opacity (float, optional): Opacity of the Voronoi diagram overlay.
            Defaults to 0.5.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay, 
        and text displaying the proportion of the field controlled by each team.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    # Proportion de contrôle pour chaque équipe
    team_1_area = np.sum(control_mask)
    team_2_area = np.sum(~control_mask)
    total_area = team_1_area + team_2_area

    team_1_percentage = (team_1_area / total_area) * 100
    team_2_percentage = (team_2_area / total_area) * 100

    # Ajouter du texte sur l'image annotée pour afficher les proportions de contrôle
    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    # Ajouter le texte légendé
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_team_1 = f"Team 1: {team_1_percentage:.2f}%"
    text_team_2 = f"Team 2: {team_2_percentage:.2f}%"

    # Positionner le texte sur l'image
    cv2.putText(overlay, text_team_1, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, text_team_2, (10, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return overlay

from typing import List, Tuple

def draw_paths_and_hull_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    hull_color: Tuple[int, int, int] = (0, 0, 255),  # Bleu pur
    path_color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    alpha: float = 0.3,  # Transparence
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths and a convex hull on a soccer pitch with semi-transparency.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        hull_color (Tuple[int, int, int], optional): Base color for the convex hull.
            Defaults to (0, 0, 255) (pure blue).
        path_color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        alpha (float, optional): Transparency level for the hull (0.0 to 1.0).
            Defaults to 0.3.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with paths and convex hull.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # Draw paths
    all_points = []
    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        all_points.extend(scaled_path)

        if len(scaled_path) < 2:
            continue

    # Draw convex hull if there are enough points
    if len(all_points) >= 3:
        hull = cv2.convexHull(np.array(all_points))
        overlay = pitch.copy()
        cv2.fillPoly(
            img=overlay,
            pts=[hull],
            color=hull_color
        )
        # Apply transparency to the hull
        cv2.addWeighted(overlay, alpha, pitch, 1 - alpha, 0, pitch)

    return pitch
