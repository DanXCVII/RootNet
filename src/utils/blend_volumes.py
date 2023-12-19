import numpy as np
import math


def linear_blend(volume1, volume2, overlap, axis) -> np.ndarray:
    """Linearly blend two volumes along a specified axis."""
    alpha = np.linspace(0, 1, overlap)

    # Select slices depending on the axis
    if axis == 0:
        blended_region = (
            alpha[:, None, None] * volume2[:overlap]
            + (1 - alpha[:, None, None]) * volume1[-overlap:]
        )
    elif axis == 1:
        blended_region = (
            alpha[None, :, None] * volume2[:, :overlap]
            + (1 - alpha[None, :, None]) * volume1[:, -overlap:]
        )
    elif axis == 2:
        blended_region = (
            alpha[None, None, :] * volume2[:, :, :overlap]
            + (1 - alpha[None, None, :]) * volume1[:, :, -overlap:]
        )
    else:
        raise ValueError("Axis must be 0, 1, or 2")

    return blended_region


def blend_volumes_to_target_size(volumes, target_size, axis, overlap) -> np.ndarray:
    """Blend a list of volumes to reach a target size along a specified axis."""
    # Initialize the output volume
    output_shape = list(volumes[0].shape)
    output_shape[axis] = target_size
    blended_volume = np.zeros(output_shape, dtype=volumes[0].dtype)

    current_position = 0

    used_indices = []
    for i, volume in enumerate(volumes):
        volume_size = volume.shape[axis]

        if i == 0:  # First volume, no blending required)
            end_position = current_position + volume_size
            slices = [slice(None)] * 3
            slices[axis] = slice(current_position, end_position)

            blended_volume[tuple(slices)] = volume
        else:
            # Blend with the previous volume
            blend_region = linear_blend(volumes[i - 1], volume, overlap, axis)
            blend_start = current_position - overlap
            blend_end = current_position
            blend_slices = [slice(None)] * 3
            blend_slices[axis] = slice(blend_start, blend_end)
            blended_volume[tuple(blend_slices)] = blend_region

            # Add the non-overlapping part
            end_position = blend_end + (volume_size - overlap)
            if end_position > target_size:
                end_position = target_size
            cut_size = end_position - blend_end + overlap
            non_overlap_slices = [slice(None)] * 3
            non_overlap_slices[axis] = slice(blend_end, end_position)
            blended_volume[tuple(non_overlap_slices)] = (
                volume[overlap:cut_size]
                if axis == 0
                else volume[:, overlap:cut_size]
                if axis == 1
                else volume[:, :, overlap:cut_size]
            )

        current_position += volume_size - overlap
        used_indices.append(i)

        if current_position >= target_size:
            break

    non_used_volumes = [
        volumes[i] for i in range(len(volumes)) if i not in used_indices
    ]

    return non_used_volumes, blended_volume


def expand_volume_with_blending(volumes, target_shape, overlap) -> np.ndarray:
    """
    Attaches the given volumes with linear blending for the specified overlap until the target shape is reached.
    Enough volumes must be given to not run out of them before the target shape is reached.

    Args:
    - volumes: List of volumes to blend together.
    - target_shape: Shape of the final volume.
    - overlap: Amount of overlap between the volumes.
    """
    volumes_shape = volumes[0].shape

    level_volumes = []
    num_blend_z = math.ceil(target_shape[2] / (volumes_shape[2] - overlap))
    for i in range(num_blend_z):
        num_blend_y = math.ceil(target_shape[0] / (volumes_shape[0] - overlap))
        expanded_volumes = []

        for j in range(num_blend_y):
            volumes, expanded_volume = blend_volumes_to_target_size(
                volumes, target_shape[1], 1, overlap
            )
            expanded_volumes.append(expanded_volume)

        _, level_volume = blend_volumes_to_target_size(
            expanded_volumes, target_shape[0], 0, overlap
        )
        level_volumes.append(level_volume)

    _, final_volume = blend_volumes_to_target_size(
        level_volumes, target_shape[2], 2, overlap
    )

    return final_volume
