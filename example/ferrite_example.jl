using WaterLily, FerriteShells, WaterLilyMeshBodies, WriteVTK

function make_cantilever(dims; L=10.0, W=1.0)
    corners = [FerriteShells.Vec{2}((0.0, 0.0)), FerriteShells.Vec{2}((L, 0.0)),
               FerriteShells.Vec{2}((L, W)),     FerriteShells.Vec{2}((0.0, W))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, dims, corners))
    return grid
end

# make a mesh
grid = make_cantilever((32,4))

# make a waterlily body
body = MeshBody(grid; map=(x,t)->x, boundary=false, half_thk=2.f0, scale=1.f0)
