using WaterLily,StaticArrays,BiotSavartBCs,WaterLilyMeshBodies
function dolphin(;scale=1f0,Re=1e6,U=1,mem=Array)
    L = round(Int,64*scale); x₀ = SA[L÷4,4+L÷2,2L÷5]
    body = MeshBody("example\\LowPolyDolphin.stl";scale,map=(x,t)->x-x₀,mem)
    BiotSimulation((L÷2,3L÷2,3L÷4),(0,U,0),L;body,T=Float32,ν=U*L/Re,mem)
end

using GLMakie,Meshing,CUDA
Makie.inline!(false)
CUDA.allowscalar(false)
dolphin_sim = dolphin(scale=3f0,mem=CUDA.CuArray); sim_step!(dolphin_sim,1;verbose=true,remeasure=false)
viz!(dolphin_sim,body2mesh=true,remeasure=false,
    azimuth=-0.5,fig_size=(1200,800),
    duration=1,step=0.01,video="dolphin.mp4",
    colormap=:ocean,colorrange=(0.15,0.5),algorithm=:mip,body_color=:white)