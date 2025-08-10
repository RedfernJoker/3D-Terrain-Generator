from PIL import Image
import numpy as np
import cv2
import math
from skimage.morphology import skeletonize
from collections import deque
from scipy.ndimage import gaussian_filter
import trimesh

# Color definitions
COLOR_BLUE = np.array([121, 183, 220])
COLOR_GREEN = np.array([171, 201, 175])
COLOR_YELLOW = np.array([213, 213, 214])

# Parameter settings
heavy_sigma = 40.0    # Pre-blur intensity
sigma_final = 10.0    # Final weak blur intensity

# Ridge sharp + diffusion parameters
ridge_sigma_sharp = 1.0
ridge_sigma_halo = 10.0
ridge_amp_sharp = 5
ridge_amp_halo = 1

# Skeleton sharp + diffusion parameters
skel_sigma_sharp = 1
skel_sigma_halo = 15.0
skel_amp_sharp = 5
skel_amp_halo = 1

# Peak Envelope (almost ignored)
sigma_peak = 8.0
h_peak = 0.1    # Very low peak intensity
w_peak = 0.1    # Peak weight

# Curve/ring peak sampling limit
max_peaks = 8
peak_interval = 30

# Noise parameters
alpha_mid = 0.05  # Mid-frequency noise intensity
alpha_high = 0.03  # High-frequency noise intensity
mid_sigma = 15
high_sigma = 3

gamma = 1.15  # Reduce gamma value to prevent mid-tones from becoming too dark
# Only keep areas above threshold unchanged
threshold = 0.9  # Lower threshold to preserve more highlight areas

# Color scale
colorscale = [
    [0.0, 'rgb(  0,   0, 128)'],  # Deep sea
    [0.2, 'rgb(  0,   0, 255)'],  # Shallow sea
    [0.3, 'rgb(  0, 128,   0)'],  # Plain
    [0.5, 'rgb( 34, 139,  34)'],  # Grassland
    [0.7, 'rgb(205, 133,  63)'],  # Mountain foot
    [0.9, 'rgb(190, 190, 190)'],  # Mountain peak
    [1.0, 'rgb(255, 255, 255)'],  # Snow peak
]


def get_mask_by_color(img, target_color, threshold=30):
    diff = np.linalg.norm(img - target_color, axis=2) if img.ndim == 3 else -1
    return diff < threshold


def compute_ridge_segmented(mask, start, init_dir, dist_map,
                            segments=10, segment_length=12,
                            inertia=0.0, grad_coef=0.15,
                            noise_std=0.5, min_turn=np.pi/36, max_turn=np.pi/12):
    path = [start]
    curr = np.array(start, float)
    direction = np.array(init_dir, float)
    if np.linalg.norm(direction) > 0:
        direction /= np.linalg.norm(direction)
    gy, gx = np.gradient(dist_map)
    for _ in range(segments):
        gdir = np.array([-gy[int(curr[0]), int(curr[1])],
                         -gx[int(curr[0]), int(curr[1])]], float)
        if np.linalg.norm(gdir) > 0:
            gdir /= np.linalg.norm(gdir)
        new_dir = inertia*direction + grad_coef*gdir + np.random.normal(scale=noise_std, size=2)
        if np.linalg.norm(new_dir) == 0:
            break
        new_dir /= np.linalg.norm(new_dir)
        cosang = np.clip(np.dot(direction, new_dir), -1, 1)
        ang = math.acos(cosang)
        sign = np.sign(direction[0]*new_dir[1] - direction[1]*new_dir[0])
        if abs(ang) < min_turn:
            ang = min_turn * sign
        elif abs(ang) > max_turn:
            ang = max_turn * sign
        c, s = math.cos(ang), math.sin(ang)
        direction = np.array([direction[0]*c - direction[1]*s,
                              direction[0]*s + direction[1]*c])
        direction /= np.linalg.norm(direction)
        for _ in range(segment_length):
            nxt = curr + direction
            iy, ix = int(round(nxt[0])), int(round(nxt[1]))
            if ix < 0 or ix >= mask.shape[1] or iy < 0 or iy >= mask.shape[0] or not mask[iy, ix]:
                return path
            curr = np.array([iy, ix], float)
            path.append((iy, ix))
    return path


def sample_peaks_on_skeleton(skel_region, shape):
    peaks = []
    H, W = skel_region.shape
    if not skel_region.any():
        return peaks
    if shape == 'curve':
        # find endpoints
        dirs8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        coords = np.argwhere(skel_region)
        ends = []
        for y, x in coords:
            deg = sum((0 <= y+dy < H and 0 <= x+dx < W and skel_region[y+dy, x+dx]) for dy, dx in dirs8)
            if deg == 1:
                ends.append((y, x))
        if len(ends) >= 2:
            pts = np.array(ends)
            d2 = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmax(d2), d2.shape)
            start, end = ends[i], ends[j]
            # BFS path
            dq = deque([start])
            prev = {start: None}
            while dq:
                cur = dq.popleft()
                if cur == end:
                    break
                cy, cx = cur
                for dy, dx in dirs8:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < H and 0 <= nx < W and skel_region[ny, nx] and (ny, nx) not in prev:
                        prev[(ny, nx)] = (cy, cx)
                        dq.append((ny, nx))
            # reconstruct
            path = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path = path[::-1]
        else:
            path = [tuple(p) for p in coords]
        L = len(path)
        n = max(1, math.ceil(L/peak_interval))
        for k in range(n):
            idx = int(k*(L-1)/(n-1)) if n > 1 else L//2
            peaks.append(path[idx])
    elif shape == 'ring':
        sk_uint = (skel_region*255).astype(np.uint8)
        cnts, _ = cv2.findContours(sk_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            loop = [(pt[0][1], pt[0][0]) for pt in cnt]
            L = len(loop)
            if L == 0:
                continue
            n = max(1, math.ceil(L/peak_interval))
            for k in range(n):
                idx = int(k*(L-1)/(n-1)) if n > 1 else L//2
                peaks.append(loop[idx])
    # clamp to max_peaks
    if len(peaks) > max_peaks:
        idxs = np.linspace(0, len(peaks)-1, max_peaks, dtype=int)
        peaks = [peaks[i] for i in idxs]
    return peaks


def extract_seeds(mask, dist):
    skel_all = skeletonize(mask.astype(bool)).astype(np.uint8)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0] if hier is not None else []
    peaks = []
    ridges = []
    gy, gx = np.gradient(dist)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ext = cv2.dilate(mask, kern)
    for i, cnt in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        # shape test
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = 4*math.pi*area/(peri*peri+1e-8)
        hullA = cv2.contourArea(cv2.convexHull(cnt))
        sol = area/(hullA+1e-8)
        is_ring = any(h[3] == i for h in hier)
        shape = 'ring' if is_ring else ('circle' if circ > 0.7 and sol > 0.8 else 'curve')
        # region minus holes
        reg = np.zeros_like(mask)
        cv2.drawContours(reg, [cnt], -1, 1, -1)
        for j, h in enumerate(hier):
            if h[3] == i:
                cv2.drawContours(reg, [cnts[j]], -1, 0, -1)
        skel_reg = skeletonize(reg.astype(bool)).astype(np.uint8)
        if shape == 'circle':
            pts = np.argwhere(skel_reg)
            if pts.size == 0:
                continue
            peak = tuple(pts[pts.shape[0]//2])
            peaks.append(peak)
            n = np.random.choice([3, 4])
            t0 = np.random.uniform(0, 2*math.pi)
            for k in range(n):
                th = t0 + 2*math.pi*k/n
                ridges.append(compute_ridge_segmented(ext, peak, (math.sin(th), math.cos(th)), dist))
        else:
            pks = sample_peaks_on_skeleton(skel_reg, shape)
            if shape == 'ring' and not pks:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cy = int(M['m01']/M['m00'])
                    cx = int(M['m10']/M['m00'])
                    pks = [(cy, cx)]
            for peak in pks:
                peaks.append(peak)
                dir0 = np.array([-gy[peak], -gx[peak]], float)
                if np.linalg.norm(dir0) > 0:
                    dir0 /= np.linalg.norm(dir0)
                else:
                    dir0 = np.array([1.0, 0.0])
                for init in (tuple(dir0), tuple(-dir0)):
                    ridges.append(compute_ridge_segmented(ext, peak, init, dist))
    return skel_all, peaks, ridges


def render_heightfield(heightmap):
    # normalize height values to [0,1] range
    h0 = heightmap / 100.0
    base = cv2.GaussianBlur(h0.astype(np.float32), (0, 0), heavy_sigma)
    mask = (h0 == h0.max()).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Darken plain by 0.025
    plain_mask = (mask == 0)
    base[plain_mask] = np.clip(base[plain_mask] - 0.025, 0, 1)

    skel, peaks, ridges = extract_seeds(mask, dist)
    H, W = h0.shape
    # ridge intensity along path
    ridge_int = np.zeros((H, W), float)
    for r in ridges:
        L = len(r)
        if L > 1:  # Add check to avoid L-1 being zero
            for i, (y, x) in enumerate(r):
                ridge_int[y, x] += 1 - i/(L-1)
    if ridge_int.max() > 0:
        ridge_int /= ridge_int.max()
    sharp_r = cv2.GaussianBlur(ridge_int.astype(np.float32), (0, 0), ridge_sigma_sharp)
    halo_r = cv2.GaussianBlur(ridge_int.astype(np.float32), (0, 0), ridge_sigma_halo)
    env_ridge = ridge_amp_sharp*sharp_r + ridge_amp_halo*halo_r
    # skeleton intensity
    dist_skel = -np.ones((H, W), float)
    dq = deque()
    for y, x in peaks:
        dist_skel[y, x] = 0
        dq.append((y, x))
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while dq:
        y, x = dq.popleft()
        for dy, dx in dirs4:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and dist_skel[ny, nx] < 0:
                dist_skel[ny, nx] = dist_skel[y, x] + 1
                dq.append((ny, nx))
    skel_int = np.zeros_like(dist_skel)
    valid = dist_skel >= 0
    if valid.any():
        max_dist = dist_skel[valid].max()
        if max_dist > 0:  # Avoid division by zero error
            skel_int[valid] = 1 - dist_skel[valid]/max_dist
        else:
            skel_int[valid] = 1  # When all skeleton points have same distance
    sharp_s = cv2.GaussianBlur(skel_int.astype(np.float32), (0, 0), skel_sigma_sharp)
    halo_s = cv2.GaussianBlur(skel_int.astype(np.float32), (0, 0), skel_sigma_halo)
    env_skel = skel_amp_sharp*sharp_s + skel_amp_halo*halo_s
    # peak envelope
    peak_int = np.zeros((H, W), float)
    for y, x in peaks:
        peak_int[y, x] = 1
    dtp = cv2.distanceTransform((1-peak_int).astype(np.uint8), cv2.DIST_L2, 5)
    env_peak = h_peak * np.exp(-dtp**2/(2*sigma_peak**2))
    # combine & final blur
    comb = base + w_peak*env_peak + env_ridge + env_skel
    comb = np.clip(comb, 0, 1)
    final = cv2.GaussianBlur(comb.astype(np.float32), (0, 0), sigma_final)
    
    # Apply color curve processing to increase contrast
    # Use power function to achieve curve effect: x^gamma
    # gamma < 1 brightens dark areas, gamma > 1 darkens areas outside bright regions

    mask = final >= threshold
    curve_applied = np.power(final, gamma)
    # Keep the whitest areas unchanged
    curve_applied[mask] = final[mask]
    final = curve_applied
    
    return final


def get_terrain_mesh(elevation_raw, fudge_factor=1.2, exponent=1.3, compress=0.8, height_scale=0.75):
    elevation_final = np.clip(elevation_raw * fudge_factor, 0, 1) ** exponent

    Z = elevation_final ** compress
    Z *= height_scale

    h, w = Z.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    return X, Y, Z


def export_terrain_to_gltf(X, Y, Z, filename="terrain1.glb"):

    h, w = Z.shape
    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            faces.append([idx, idx + 1, idx + w])
            faces.append([idx + 1, idx + w + 1, idx + w])
    faces = np.array(faces)
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Export as glTF
    mesh.export(filename)
    print(f"Terrain exported as {filename}")


def process_image_to_terrain(image_path, output_path):
    try:
        print(f"Processing image: {image_path}")
        
        # read image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        print(f"Image size: {image_np.shape}")
        
        # generate heightmap
        blue_mask = get_mask_by_color(image_np, COLOR_BLUE)
        green_mask = get_mask_by_color(image_np, COLOR_GREEN)
        yellow_mask = get_mask_by_color(image_np, COLOR_YELLOW)

        heightmap = np.full(image_np.shape[:2], 40, dtype=np.float32)   # Plain
        heightmap[blue_mask] = 0                                       # Ocean
        heightmap[yellow_mask] = 95                                    # Mountain
        
        # render terrain directly
        base_hf = render_heightfield(heightmap)
        
        # add noise
        np.random.seed(123)
        noise = np.random.randn(image_np.shape[0], image_np.shape[1])
        mid_noise = gaussian_filter(noise, sigma=mid_sigma)
        high_noise = gaussian_filter(noise, sigma=high_sigma)
        
        # normalize noise 
        epsilon = 1e-8
        mid_range = mid_noise.max() - mid_noise.min()
        high_range = high_noise.max() - high_noise.min()
        
        if mid_range > epsilon:
            mid_noise = (mid_noise - mid_noise.min()) / mid_range
        else:
            mid_noise = np.zeros_like(mid_noise)
            
        if high_range > epsilon:
            high_noise = (high_noise - high_noise.min()) / high_range
        else:
            high_noise = np.zeros_like(high_noise)
        
        # mix noise with base height
        combined = base_hf + alpha_mid * mid_noise + alpha_high * high_noise
        combined = np.clip(combined, 0, 1)
        
        # generate 3D terrain
        X, Y, Z = get_terrain_mesh(combined, fudge_factor=1, exponent=1.3)
        
        # export to glTF format
        export_terrain_to_gltf(X, Y, Z, filename=output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Terrain generation error: {str(e)}")
        raise e


if __name__ == "__main__":
    input_path = "input.png"  
    output_path = "terrain_output.glb"  
    
    process_image_to_terrain(input_path, output_path)