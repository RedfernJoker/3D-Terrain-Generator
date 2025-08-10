import { useState, useRef, useEffect } from 'react'
import './App.css'
import Render from './Render'

function App() {
  // props
  const canvasWidth = 450;
  const canvasHeight = 450;

  const colorPalette = {
    mountain: "#D5D5D6",
    valley: "#ABC9AF",
    ocean: "#79B7DC",
    clean: "#FFFFFF"
  }

  const minBrushSize = 30;
  const maxBrushSize = 80;
  
  // state
  const [selectedBrush, setSelectedBrush] = useState<string>("mountain");
  const [brushSize, setBrushSize] = useState<number>(minBrushSize);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const [terrainUrl, setTerrainUrl] = useState<string>("/models/t1.glb");

  // setup canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    const context = canvas.getContext('2d');
    if (!context) return;
    
    // line style 
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.strokeStyle = getBrushColor(selectedBrush);
    context.lineWidth = brushSize;
    
    // fill background 
    context.fillStyle = "#FFFFFF";
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    contextRef.current = context;
  }, []);
  
  // change brush  
  useEffect(() => {
    if (!contextRef.current) return;

    contextRef.current.strokeStyle = getBrushColor(selectedBrush);
    contextRef.current.lineWidth = brushSize;

  }, [selectedBrush, brushSize]);
  
  // align brush color
  const getBrushColor = (brushType: string): string => {
    return colorPalette[brushType as keyof typeof colorPalette];
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    contextRef.current.beginPath();
    contextRef.current.moveTo(x, y);
    setIsDrawing(true);
  };
  
  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !contextRef.current || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    contextRef.current.lineTo(x, y);
    contextRef.current.stroke();
  };
  
  const stopDrawing = () => {
    if (!contextRef.current) return;
    contextRef.current.closePath();
    setIsDrawing(false);
  };

  const handleBrushSelect = (brushType: string) => {
    setSelectedBrush(brushType);
  };

  const handleBrushSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBrushSize(parseInt(e.target.value));
  };

  const resetCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;
    
    contextRef.current.fillStyle = "#FFFFFF";
    contextRef.current.fillRect(0, 0, canvasWidth, canvasHeight);
  };


  const processCanvasColors = (canvas: HTMLCanvasElement, targetColorHex: string = "#ABC9AF"): string => {
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (!tempCtx) return canvas.toDataURL('image/png');
    
    tempCtx.drawImage(canvas, 0, 0);
    
    const imageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
    const pixelData = imageData.data;
    const width = canvas.width;
    const height = canvas.height;
    
    const r = parseInt(targetColorHex.slice(1, 3), 16);
    const g = parseInt(targetColorHex.slice(3, 5), 16);
    const b = parseInt(targetColorHex.slice(5, 7), 16);
    
    const pixelsToReplace: boolean[] = new Array(pixelData.length / 4).fill(false);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        if (pixelData[idx] > 240 && pixelData[idx+1] > 240 && pixelData[idx+2] > 240) {
          pixelsToReplace[y * width + x] = true;
        }
      }
    }
    
    const expandedPixels = [...pixelsToReplace];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (pixelsToReplace[y * width + x]) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const ny = y + dy;
              const nx = x + dx;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                expandedPixels[ny * width + nx] = true;
              }
            }
          }
        }
      }
    }
    
    for (let i = 0; i < expandedPixels.length; i++) {
      if (expandedPixels[i]) {
        const idx = i * 4;
        pixelData[idx] = r;
        pixelData[idx+1] = g;
        pixelData[idx+2] = b;
      }
    }
    
    tempCtx.putImageData(imageData, 0, 0);
    
    return tempCanvas.toDataURL('image/png');
  };

  const handleConvert = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      const processedImageData = processCanvasColors(canvas);
      
      // Convert data URL to blob
      const response = await fetch(processedImageData);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append('file', blob, 'terrain.png');
      
      const apiResponse = await fetch('http://localhost:8000/generate-terrain', {
        method: 'POST',
        body: formData,
      });
      
      if (!apiResponse.ok) {
        throw new Error('server response error');
      }
      
      // get the model file name 
      const responseData = await apiResponse.json();

      console.log(responseData.modelFileName) 

      setTerrainUrl(`/models/${responseData.modelFileName}?t=${Date.now()}`);
    } catch (error) {
      console.error('terrain generation failed:', error);
    }
  };



  return (
    <div className="bg-[#F5F5F5] min-h-screen">
      <main className="container mx-auto h-screen p-4 montserrat-SemiBold flex flex-col justify-center items-center">
        <div className="flex justify-center mb-6">
          <h1 className="text-4xl font-bold text-center">3D Terrain Generator</h1>
        </div>
        
        <div id="drawing-section" className="grid grid-cols-2 gap-6 justify-center w-[900px]">
          
          <div id="canvas-container" className="w-[450px] h-[450px] border-4 border-black rounded-xl overflow-hidden">
            <canvas 
              id="mapCanvas" 
              ref={canvasRef}
              className="w-full h-full"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            ></canvas>
          </div>

          <div id="render-container" className="w-[450px] h-[450px] border-4 border-black rounded-xl overflow-hidden">
            <Render terrainUrl={terrainUrl} />
          </div>
          
          <div id="tool-container" className="text-sm col-span-2 w-full rounded-md flex justify-between items-center mt-4">
            <div id="tool-bars" className="flex gap-6 items-center">
              <div id="brush-bar" className="flex items-center gap-6 tracking-tight">
                <div 
                  className={`flex flex-col items-center cursor-pointer`} 
                  onClick={() => handleBrushSelect("mountain")}
                >
                  <div className={`mb-1 p-1 rounded-full ${selectedBrush === "mountain" ? `ring-2 ring-[#D5D5D6]` : ""}`}>
                    <div className="w-12 h-12 rounded-full bg-[#D5D5D6]"></div>
                  </div>
                  <span className="montserrat-Bold text-sm">Mountain</span>
                </div>

                <div 
                  className={`flex flex-col items-center cursor-pointer`}
                  onClick={() => handleBrushSelect("valley")}
                > 
                  <div className={`mb-1 p-1 rounded-full ${selectedBrush === "valley" ? `ring-2 ring-[#ABC9AF]` : ""}`}>
                    <div className={`w-12 h-12 rounded-full bg-[#ABC9AF] `}></div>
                  </div>
                  <span className="montserrat-Bold text-sm">Plains</span>
                </div>

                <div 
                  className={`flex flex-col items-center cursor-pointer`}
                  onClick={() => handleBrushSelect("ocean")}
                >
                  <div className={`mb-1 p-1 rounded-full ${selectedBrush === "ocean" ? `ring-2 ring-[#79B7DC]` : ""}`}>
                    <div className={`w-12 h-12 rounded-full bg-[#79B7DC]`}></div>
                  </div>
                  <span className="montserrat-Bold text-sm">Ocean</span>
                </div>

                <div 
                  className={`flex flex-col items-center cursor-pointer`}
                  onClick={() => handleBrushSelect("clean")}
                >
                  <div className={`mb-1 p-1 rounded-full ${selectedBrush === "clean" ? `ring-2 ring-[#FFC1C1]` : ""}`}>
                    <div className={`w-12 h-12 rounded-full bg-[#FFC1C1]`}></div>
                  </div>
                  <span className="montserrat-Bold text-sm">Clean</span>
                </div>
              </div>

              <div id="slider-bar" className="px-3">
                <div className="flex items-center gap-2">
                  <input 
                    type="range" 
                    min={minBrushSize} 
                    max={maxBrushSize} 
                    value={brushSize} 
                    onChange={handleBrushSizeChange}
                    className="w-32 h-[6px] bg-black rounded-lg appearance-none cursor-pointer slider-thumb" 
                    style={{
                      '--thumb-color': `${selectedBrush === "clean" ? "#CBCBCB" : getBrushColor(selectedBrush)}`
                    } as React.CSSProperties}
                  />
                </div>
              </div>
            </div>

            <div id="action-bars" className="text-sm flex gap-4 items-center montserrat-Bold tracking-tight">
              <button 
                className="w-32 px-4 py-3 bg-transparent text-black border-2 border-black rounded-full hover:bg-gray-200 hover:cursor-pointer text-sm"
                onClick={resetCanvas}
              >
                Reset
              </button>
              
              <button 
                className="w-55 px-4 py-3 bg-transparent text-black border-2 border-black rounded-full hover:bg-gray-200 hover:cursor-pointer text-sm"
                onClick={handleConvert}
              >
                Generate 3D Terrain
              </button>
            </div>
          </div>
        </div>

      </main>
    </div>
  )
}

export default App
