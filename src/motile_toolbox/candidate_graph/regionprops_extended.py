from skimage.measure import mesh_surface_area, marching_cubes, regionprops
from skimage.measure._regionprops import RegionProperties
import numpy as np
import math
import scipy.ndimage                as spim
from skimage.morphology             import ball


class ExtendedRegionProperties(RegionProperties):
    """Adding additional properties to skimage.measure._regionprops following the logic from the porespy package with some modifications to include the spacing information."""

    @property
    def axes(self):
        """Calculate the three axes radii"""
        cell        = np.where(self._label_image == self.label) # np.where returns a tuple of all the indices in the different dimensions z,y,x that fulfill the condition. The indices in z are different than in y, but the length is the same.
        voxel_count = self.voxel_count

        z, y, x     = cell

        # finding the center z,y,x and calibrate
        z = (z - np.mean(z)) * self._spacing[0]
        y = (y - np.mean(y)) * self._spacing[1]
        x = (x - np.mean(x)) * self._spacing[2]


        i_xx    = np.sum(y ** 2 + z ** 2)
        i_yy    = np.sum(x ** 2 + z ** 2)
        i_zz    = np.sum(x ** 2 + y ** 2) # Moments of inertia with respect to the x, y, z, axis.
        i_xy    = np.sum(x * y)
        i_xz    = np.sum(x * z)
        i_yz    = np.sum(y * z) # Products of inertia. A measure of imbalance in the mass distribution.

        i       = np.array([[i_xx, -i_xy, -i_xz], [-i_xy, i_yy, -i_yz], [-i_xz, -i_yz, i_zz]]) # Tensor of inertia. For calculating the Principal Axes of Inertia (eigvec & eigval).
        eig     = np.linalg.eig(i)

        eigval  = eig[0]

        longaxis  = np.where(np.min(eigval) == eigval)[0][0]
        shortaxis = np.where(np.max(eigval) == eigval)[0][0]
        midaxis   = 0 if shortaxis != 0 and longaxis != 0 else 1 if shortaxis != 1 and longaxis != 1 else 2

        # Calculate 3 radii (or 3 principal axis lengths of the fitted ellipsoid.)
        longr     = math.sqrt(5.0 / 2.0 * (eigval[midaxis]   + eigval[shortaxis] - eigval[longaxis])  / voxel_count)
        midr      = math.sqrt(5.0 / 2.0 * (eigval[shortaxis] + eigval[longaxis]  - eigval[midaxis])   / voxel_count)
        shortr    = math.sqrt(5.0 / 2.0 * (eigval[longaxis]  + eigval[midaxis]   - eigval[shortaxis]) / voxel_count)
        return (longr, midr, shortr) # return calibrated three axis radii

    @property
    def circularity(self):
        return 4 * math.pi * self.area / self.perimeter**2

    @property
    def pixel_count(self):
        return self.voxel_count
    
    @property
    def surface_area(self):
        verts, faces, _, _ = marching_cubes(self._label_image == self.label, level=0.5, spacing=self._spacing)
        surface_area = mesh_surface_area(verts, faces)
        return surface_area

    @property
    def surface_area_smooth(self):
        mask = self.mask
        tmp = np.pad(np.atleast_3d(mask), pad_width=1, mode='constant')
        kernel_radii = np.array(self._spacing)
        tmp = spim.convolve(tmp, weights=ball(min(kernel_radii))) / 5  # adjust kernel size for anisotropy
        verts, faces, _, _ = marching_cubes(volume=tmp, level=0, spacing = self._spacing)
        area = mesh_surface_area(verts, faces)
        return area

    @property
    def sphericity(self):
        vol = self.volume
        r = (3 / 4 / np.pi * vol)**(1 / 3)
        a_equiv = 4 * np.pi * r**2
        a_region = self.surface_area
        return a_equiv / a_region

    @property
    def volume(self):
        vol = np.sum(self._label_image == self.label) * np.prod(self._spacing)
        return vol

    @property
    def voxel_count(self):
        voxel_count = np.sum(self._label_image == self.label)
        return voxel_count

def regionprops_extended(img: np.ndarray, spacing: tuple[float], intensity_image: np.ndarray | None = None) -> list[ExtendedRegionProperties]:
    """Create instance of ExtendedRegionProperties that extends skimage.measure.RegionProperties"""

    results = regionprops(img, intensity_image=intensity_image, spacing = spacing)
    for i, _ in enumerate(results):
        a = results[i]
        b = ExtendedRegionProperties(slice = a.slice,
                            label = a.label,
                            label_image = a._label_image,
                            intensity_image = a._intensity_image,
                            cache_active = a._cache_active,
                            spacing = a._spacing)
        results[i] = b

    return results
