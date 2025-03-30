const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});
app.get('/api/data', (req, res) => {
    res.sendFile(path.join(__dirname, 'data.json'));
});
app.get('/api/model', (req, res) => {
    res.sendFile(path.join(__dirname, 'model.json'));
});
app.get('/api/weights', (req, res) => {
    res.sendFile(path.join(__dirname, 'weights.bin'));
});
app.get('/api/metadata', (req, res) => {
    res.sendFile(path.join(__dirname, 'metadata.json'));
});
app.get('transflow.js', (req, res) => {
    res.sendFile(path.join(__dirname, 'transflow.js'));
});
app.get('transflow.css', (req, res) => {
    res.sendFile(path.join(__dirname, 'transflow.css'));
});
app.get('labels.json', (req, res) => {  
    res.sendFile(path.join(__dirname, 'labels.json'));
});
app.get('js/plotly-latest.min.js', (req, res) => {
    res.sendFile(path.join(__dirname, 'js/plotly-latest.min.js'));
});


app.post('/train', (req, res) => {
    const { data } = req.body;
    if (!data) {
        return res.status(400).send('Data untuk pelatihan tidak ditemukan.');
    }
 
    console.log('Data pelatihan diterima:', data);
    res.send('Pelatihan AI berhasil dilakukan.');
});


app.listen(PORT, () => {
    console.log(`Server pelatihan AI berjalan di http://localhost:${PORT}`);
});