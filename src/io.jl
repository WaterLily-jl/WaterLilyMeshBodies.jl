# VTK export functions

import WaterLily: AbstractBody, SetBody, save!
import WriteVTK: MeshCell, VTKCellTypes, vtk_grid, vtk_save
using GeometryBasics: TriangleFace
using Printf: @sprintf

"""

    save!(writer::VTKWriter, body::MeshBody, t=writer.count[1])

Saves the mesh body as a VTK file using the WriteVTK package. The file name is generated using the writer's directory name, base file name, and the current count.
"""
function save!(w,a::MeshBody,t=w.count[1])
    k = w.count[1]
    points = zeros(Float32, 3, 3length(a.mesh))
    for (i,el) in enumerate(Array(a.mesh))
        points[:,3i-2:3i] = el
    end
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, TriangleFace{Int}(3i+1,3i+2,3i+3)) for i in 0:length(a.mesh)-1]
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), points, cells)
    for (name,func) in w.output_attrib
        # point/vector data must be oriented in the same way as the mesh
        vtk[name] = ndims(func(a))==1 ? func(a) : permutedims(func(a))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(t,digits=4)]=vtk
end
save!(w,a::AbstractBody,t) = nothing
save!(w,a::SetBody,t) = (save!(w,a.a,t); save!(w,a.b,t))
