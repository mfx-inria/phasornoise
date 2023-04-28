"""Implementation of phasor noise with PyOpenCL."""

import os

import numpy
import pyopencl as pcl
import vtk
from vtk.util import numpy_support


def init_opencl(filename: str):
    """
    Initialize PyOpenCL and compile kernels.

    Args:
        filename: file containing the OpenCL code.

    Returns:
        Program, context and queue (if found).

    Raises:
        RuntimeError: if the compilation of the OpenCL program fails.
    """
    # select by default an Nvidia platform if available
    target_platform = pcl.get_platforms()[0]
    for platform in pcl.get_platforms():
        if 'NVIDIA' in platform.get_info(pcl.platform_info.VENDOR):
            target_platform = platform
    print(target_platform)
    if target_platform is not None:
        cl_context = pcl.Context(
            dev_type=pcl.device_type.ALL,
            properties=[(pcl.context_properties.PLATFORM, target_platform)],
        )
        with open(filename) as filecl:
            program_str = filecl.read()
            return (
                pcl.Program(cl_context, program_str).build(),
                cl_context,
                pcl.CommandQueue(cl_context),
            )
    raise RuntimeError('failed to initialize OpenCL')


def phasor_noise(
    cl_program: pcl.Program,
    cl_context: pcl.Context,
    cl_queue: pcl.CommandQueue,
    resolution: numpy.ndarray,
    frequency: float,
    number_cells: int,
    phasor_density: float,
    factor_angle_spread: float,
    make_periodic: bool,
    seed: int,
) -> numpy.ndarray:
    """
    d-dimensional phasor noise (d=2, d=3).

    Convention: the dimensions are normalized to [0, 1]
    with respect to the first dimension N1.

    Args:
        cl_program: pyopencl program.
        cl_context: pyopencl context.
        cl_queue: pyopencl queue.
        resolution: size of the grid [resolution]^d.
        frequency: number of oscillations along N1.
        number_cells: number of gabor kernels along N1: less kernels => more regularity.
        phasor_density: target density of the phasor field [0, 1].
        factor_angle_spread: distribution of angles [0: bilobe, 1:isotropic].
        make_periodic: enforce periodicity.
        seed: random seed.

    Returns:
        Grid containing the phasor noise.

    Raises:
        TypeError: if some parameter has an invalid type.
        ValueError: if some parameter has an invalid range.
    """
    if not isinstance(resolution, numpy.ndarray):
        raise TypeError('resolution must be ndarray')
    if len(resolution) not in {2, 3}:
        raise TypeError('resolution must of dimension 2 or 3')
    if numpy.any(resolution <= 0):
        raise ValueError('resolution must strictly positive')
    if not (isinstance(number_cells, int) and number_cells > 0):
        raise TypeError('number_cells must be a positive integer')
    if frequency <= 0:
        raise ValueError('frequency must be positive')
    if not (0 <= phasor_density <= 1):
        raise ValueError('phasor_density must be [0,1]')
    if not (0 <= factor_angle_spread <= 1):
        raise ValueError('factor_angle_spread must be [0,1]')
    if make_periodic:
        # periodicity requires equal resolution in all dimensions
        if not numpy.all(resolution == resolution[0]):
            raise ValueError('resolution values must be equal (periodicity)')
    # bandwidth is defined by integer number of kernels across N1
    truncate = 0.01
    log_truncate = numpy.log(truncate)
    bandwidth = number_cells * (2 * numpy.sqrt(-log_truncate / numpy.pi))
    if len(resolution) == 2:
        kernel = cl_program.phasor2D
    elif len(resolution) == 3:
        kernel = cl_program.phasor3D
    grid = numpy.zeros(resolution, dtype=numpy.float32)
    grid_buffer = pcl.Buffer(cl_context, pcl.mem_flags.WRITE_ONLY, grid.nbytes)
    cl_event = kernel(
        cl_queue,
        grid.shape,
        None,
        grid_buffer,
        pcl.cltypes.float(frequency),
        pcl.cltypes.float(bandwidth),
        pcl.cltypes.float(phasor_density),
        pcl.cltypes.float(factor_angle_spread),
        pcl.cltypes.int(make_periodic),
        pcl.cltypes.float(truncate),
        pcl.cltypes.int(seed),
    )
    cl_event.wait()
    pcl.enqueue_copy(cl_queue, grid, grid_buffer, is_blocking=True)
    return grid


def export_grid(grid, filename):
    """
    Export phasor noise (dimension 2 or 3) to VTK image data format (vti).

    Args:
        grid: numpy array containing the grid data.
        filename: output filename (vti)
    """
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=grid.flatten(order='F'),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_img = vtk.vtkImageData()
    vtk_img.GetPointData().SetScalars(vtk_data)
    if grid.ndim == 2:
        vtk_img.SetDimensions(
            grid.shape[0], grid.shape[1], 1,
        )
    elif grid.ndim == 3:
        vtk_img.SetDimensions(
            grid.shape[0], grid.shape[1], grid.shape[2],
        )
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_img)
    writer.Write()


if __name__ == '__main__':
    filename = os.path.join(os.path.dirname(__file__), 'phasor.cl')
    (cl_program, cl_context, cl_queue) = init_opencl(filename)

    # common parameters
    frequency = 20
    number_cells = 5
    phasor_density = 0.5
    seed = 42

    # 2D test (non-periodic)
    resolution2d = numpy.array([563, 977])  # (N1, N2)
    phasor2dtest = phasor_noise(
        cl_program,
        cl_context,
        cl_queue,
        resolution2d,
        frequency,
        number_cells,
        phasor_density,
        factor_angle_spread=0,
        make_periodic=False,
        seed=seed,
    )
    print('Target density (2D) = {0}'.format(phasor_density))
    print('Result density (2D) = {0}'.format(numpy.mean(phasor2dtest)))
    export_grid(phasor2dtest, 'phasor2dtest.vti')

    # 3D test (periodic)
    resolution3d = numpy.array([128, 128, 128])  # (N1, N2, N3)
    phasor3dtest = phasor_noise(
        cl_program,
        cl_context,
        cl_queue,
        resolution3d,
        frequency,
        number_cells,
        phasor_density,
        factor_angle_spread=0,
        make_periodic=True,
        seed=seed,
    )
    print('Target density (3D) = {0}'.format(phasor_density))
    print('Result density (3D) = {0}'.format(numpy.mean(phasor3dtest)))
    export_grid(phasor3dtest, 'phasor3dtest.vti')
