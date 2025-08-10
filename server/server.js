import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// ES模块中获取__dirname的替代方案
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ?? 3000;

// Enable CORS 
app.use(cors({
  origin: ['http://localhost:5173', 'http://127.0.0.1:5173'],
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  credentials: true
}));

// 添加JSON解析中间件
app.use(express.json({ limit: '50mb' }));

// 静态文件服务
app.use('/models', express.static(path.join(__dirname, '../models')));

app.get('/', (req, res) => {
  res.json({ message: 'Hello World' });
});

// sample JSON API
app.get('/ping', (req, res) => {
  res.json({ ok: true, ts: Date.now() });
});

// Health check endpoint that proxies to Python service
app.get('/health', (req, res) => {
  (async () => {
    try {
      const pyRes = await fetch('http://localhost:8000/health');
      if (!pyRes.ok) {
        return res.status(502).send('Python service error');
      }
      const data = await pyRes.json();
      res.json(data);
    } catch (err) {
      console.error(err);
      res.status(500).send('Internal server error');
    }
  })();
});

// generate terrain
app.post('/generate-terrain', (req, res) => {
  (async () => {
    try {
      const { imageData } = req.body;
      if (!imageData) {
        return res.status(400).json({ error: 'missing image data' });
      }

      // convert base64 to file
      const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
      const imageFilePath = path.join(__dirname, '../temp', `terrain_.png`);
      
      const dir = path.dirname(imageFilePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      fs.writeFileSync(imageFilePath, base64Data, 'base64');

      const pyRes = await fetch('http://localhost:8000/generate-terrain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ imagePath: imageFilePath }),
      });

      if (!pyRes.ok) {
        throw new Error('Python service response error');
      }

      const data = await pyRes.json();

      // 返回模型URL
      res.json({ modelFileName: data.modelFileName });
    } catch (error) {
      console.error('terrain generation failed:', error);
      res.status(500).json({ error: 'terrain generation failed' });
    }
  })();
});

// generate terrain image
app.post('/generate-terrain-image', (req, res) => {
  (async () => {
    try {
      const { imageData } = req.body;
      if (!imageData) {
        return res.status(400).json({ error: 'missing image data' });
      }

      // convert base64 to file
      const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
      const imageFilePath = path.join(__dirname, '../temp', `terrain_.png`);
      
      // ensure the directory exists
      const dir = path.dirname(imageFilePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      // save the image file
      fs.writeFileSync(imageFilePath, base64Data, 'base64');

      // call the python service to generate the terrain image
      const pyRes = await fetch('http://localhost:8000/generate-terrain-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ imagePath: imageFilePath }),
      });

      if (!pyRes.ok) {
        throw new Error('Python service response error');
      }

      const data = await pyRes.json();

      // return the image file name
      res.json({ imageFileName: data.imageFileName });
    } catch (error) {
      console.error('terrain generation failed:', error);
      res.status(500).json({ error: 'terrain generation failed' });
    }
  })();
});


app.listen(PORT, () => console.log(`http://localhost:${PORT}`));