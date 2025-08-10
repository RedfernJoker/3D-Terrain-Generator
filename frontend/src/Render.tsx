import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF  } from "@react-three/drei";
import { Suspense, useLayoutEffect } from "react";
import * as THREE from "three";

function Terrain({ url }: { url: string }) {
  // load the .glb file 
  const { scene } = useGLTF(url) as any;              
  scene.rotation.x = -Math.PI / 2

  const getColor = (t: number) => {    
    t = t / 0.75;
    if (t < 0.2) {
      return new THREE.Color("rgb(0, 0, 128)").lerp(new THREE.Color("rgb(0, 128, 192)"), t / 0.2);
    } else if (t < 0.25) {
      return new THREE.Color("rgb(0, 128, 192)").lerp(new THREE.Color("rgb(0, 100, 34)"), (t - 0.2) / 0.05);
    } else if (t < 0.4) {
      return new THREE.Color("rgb(0, 100, 34)").lerp(new THREE.Color("rgb(34, 139, 34)"), (t - 0.25) / 0.15);
    } else if (t < 0.7) {
      return new THREE.Color("rgb(34, 139, 34)").lerp(new THREE.Color("rgb(222, 184, 135)"), (t - 0.4) / 0.3);
    } else if (t < 0.9) {
      return new THREE.Color("rgb(222, 184, 135)").lerp(new THREE.Color("rgb(190, 190, 190)"), (t - 0.7) / 0.2);
    } else {
      return new THREE.Color("rgb(190, 190, 190)").lerp(new THREE.Color("rgb(255, 255, 255)"), (t - 0.9) / 0.1);
    }
  }

  // create the closed terrain
  const createSides = (mesh: THREE.Mesh) => {
    const geometry = mesh.geometry as THREE.BufferGeometry;
    const positions = geometry.attributes.position.array;
    const vertexCount = geometry.attributes.position.count;
    
    const minX = 0;
    const maxX = 1;
    const minY = 0;
    const maxY = 1;
    const baseZ = 0; 

    // find the edge points
    const edgePoints = {
      minX: [] as THREE.Vector3[],
      maxX: [] as THREE.Vector3[],
      minY: [] as THREE.Vector3[],
      maxY: [] as THREE.Vector3[]
    };
    
    // epsilon
    const epsilon = 0.001; 
    
    for (let i = 0; i < vertexCount; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      const point = new THREE.Vector3(x, y, z);
      
      if (Math.abs(x - minX) < epsilon) edgePoints.minX.push(point.clone());
      if (Math.abs(x - maxX) < epsilon) edgePoints.maxX.push(point.clone());
      if (Math.abs(y - minY) < epsilon) edgePoints.minY.push(point.clone());
      if (Math.abs(y - maxY) < epsilon) edgePoints.maxY.push(point.clone());
    }
    
    // create the side
    const createSide = (edgePoints: THREE.Vector3[], isXAxis: boolean) => {
      if (edgePoints.length < 2) return null;
      
      // sort the edge points
      edgePoints.sort((a, b) => {
        return isXAxis ? a.y - b.y : a.x - b.x;
      });
      
      const sideGeo = new THREE.BufferGeometry();
      const vertices = [];
      const colors = [];
      
      // create the side
      for (let i = 0; i < edgePoints.length; i++) {
        const topPoint = edgePoints[i];
        const bottomPoint = topPoint.clone();
        bottomPoint.z = baseZ;
        
        if (i < edgePoints.length - 1) {
          const nextTop = edgePoints[i + 1];
          const nextBottom = nextTop.clone();
          nextBottom.z = baseZ;
          
          vertices.push(
            topPoint.x, topPoint.y, topPoint.z,
            bottomPoint.x, bottomPoint.y, bottomPoint.z,
            nextTop.x, nextTop.y, nextTop.z,
            
            nextTop.x, nextTop.y, nextTop.z,
            bottomPoint.x, bottomPoint.y, bottomPoint.z,
            nextBottom.x, nextBottom.y, nextBottom.z
          );
          
          // add the color 
          const topColor = getColor(topPoint.z);
          const bottomColor = getColor(bottomPoint.z); 
          
          colors.push(
            topColor.r, topColor.g, topColor.b,
            bottomColor.r, bottomColor.g, bottomColor.b,
            topColor.r, topColor.g, topColor.b,
            
            topColor.r, topColor.g, topColor.b,
            bottomColor.r, bottomColor.g, bottomColor.b,
            bottomColor.r, bottomColor.g, bottomColor.b
          );
        }
      }
      
      sideGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      sideGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      sideGeo.computeVertexNormals();
      
      return sideGeo;
    };
    
    // create the four side
    const sideXMin = createSide(edgePoints.minX, false);
    const sideXMax = createSide(edgePoints.maxX, false);
    const sideYMin = createSide(edgePoints.minY, true);
    const sideYMax = createSide(edgePoints.maxY, true);
    
    // create the bottom
    const baseSide = new THREE.PlaneGeometry(1, 1, 1, 1);
    baseSide.translate((maxX + minX) / 2, (maxY + minY) / 2, baseZ);
    
    const baseColor = getColor(baseZ);
    const baseColors = [];
    for (let i = 0; i < baseSide.attributes.position.count; i++) {
      baseColors.push(baseColor.r, baseColor.g, baseColor.b);
    }
    baseSide.setAttribute('color', new THREE.Float32BufferAttribute(baseColors, 3));
    
    // create the material
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.8,
      metalness: 0.1,
      side: THREE.DoubleSide
    });
    
    // add the side and bottom
    if (sideXMin) scene.add(new THREE.Mesh(sideXMin, material));
    if (sideXMax) scene.add(new THREE.Mesh(sideXMax, material));
    if (sideYMin) scene.add(new THREE.Mesh(sideYMin, material));
    if (sideYMax) scene.add(new THREE.Mesh(sideYMax, material));
    scene.add(new THREE.Mesh(baseSide, material));
  };

  // create the scene
  useLayoutEffect(() => {
    // get the mesh 
    const mesh = scene.getObjectByProperty("type", "Mesh") as THREE.Mesh;
    if (!mesh) return;

    const geom = mesh.geometry as THREE.BufferGeometry;
    const pos   = geom.attributes.position as THREE.BufferAttribute;
    const count = pos.count;

    // generate the color array
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const t = pos.getZ(i);
      const tempColor = getColor(t);
      colors.set([tempColor.r, tempColor.g, tempColor.b], i * 3);
    }
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.attributes.color.needsUpdate = true;

    // replace the material
    mesh.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.8,  
      metalness: 0.1,  
    });

    // close the terrain
    createSides(mesh);
  
  }, [scene]);
  
  return <primitive object={scene} dispose={null} />;
}

function Sea({ width, depth, level = 0 }: { width: number, depth: number, level: number } ) {
  const waterDepth = 0.19;
  const offset = 0.0001;
  
  return (
    <>
      <mesh position={[0.5, level - waterDepth/2, -0.5]}>
        <boxGeometry args={[width - offset, waterDepth - offset, depth - offset]} />
        <meshPhysicalMaterial
          transparent
          opacity={0.7}
          roughness={0.2}
          metalness={0.1}
          clearcoat={1}
          clearcoatRoughness={0.1}
          envMapIntensity={0.7}
          color="#1a5db6"
          transmission={0.4}
          thickness={0.5}
        />
      </mesh>
    </>
  );
}

function Render({ terrainUrl }: { terrainUrl: string }) {
  return (
    <Canvas 
      className="w-full h-full" 
      camera={{ 
        position: [1.1, 0.8, -1.9],  
        fov: 60,               
        near: 0.1,             
        far: 100,              
        zoom: 1,               
      }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[1, 1, -1]} intensity={1.3} castShadow />
      <Suspense fallback={null}>
        <Terrain url={terrainUrl} />
        <Sea width={1} depth={1} level={0.19} />
      </Suspense>
      <OrbitControls 
        makeDefault 
        target={[0, 0, 0]}  
        enableDamping={true}  
        dampingFactor={0.05}  
        rotateSpeed={0.8}    
        zoomSpeed={1.2}      
        panSpeed={0.8}         
        minDistance={0.5}      
        maxDistance={10}       
      />
    </Canvas>
  )
}

export default Render;