/* global THREE */
function safeRequire (path) {
  try {
    return require(path)
  } catch (e) {
    return {}
  }
}
global.THREE = require('three')
global.Worker = require('worker_threads').Worker
const { createCanvas } = safeRequire('node-canvas-webgl/lib')

const fs = require('fs'); // add by yxhxianyu

const { WorldView, Viewer } = require('../viewer')

module.exports = (bot, views, { viewDistance = 6, interval = 250, width = 256, height = 144, jpegOptions }) => {
  const canvas = createCanvas(width, height)
  const renderer = new THREE.WebGLRenderer({ canvas })
  const viewer = new Viewer(renderer)

  viewer.setVersion(bot.version)
  viewer.setFirstPersonCamera(bot.entity.position, bot.entity.yaw, bot.entity.pitch)

  // Load world
  const worldView = new WorldView(bot.world, viewDistance, bot.entity.position)
  viewer.listen(worldView)
  worldView.init(bot.entity.position)

  function botPosition () {
    viewer.setFirstPersonCamera(bot.entity.position, bot.entity.yaw, bot.entity.pitch)
    worldView.updatePosition(bot.entity.position)
  }

  // Render loop streaming

  let idx = 0
  update()

  // Force end of stream
  bot.on('end', () => { frames = 0 })

  function update () {
    viewer.update()
    renderer.render(viewer.scene, viewer.camera)

    const imageBuffer = canvas.toBuffer('image/jpeg', jpegOptions);
    views[bot.username] = imageBuffer.toString('base64');

    // saveBase64AsImage(base64Image, 'output' + idx + '.jpg');

    idx++;
    setTimeout(update, interval);
  }

  function saveBase64AsImage(base64String, filename) {
    const base64Image = base64String.split(';base64,').pop();
    const imageBuffer = Buffer.from(base64Image, 'base64');
    fs.writeFile(filename, imageBuffer, {encoding: 'base64'}, function(err) {
      if (err) { console.error(err); } else { console.log('文件保存成功'); }
    });
  }

  // Register events
  bot.on('move', botPosition)
  worldView.listenToBot(bot)

  return { viewer, worldView, views }
}
