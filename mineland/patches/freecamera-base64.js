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
const TWEEN = require('@tweenjs/tween.js')
const { createCanvas } = safeRequire('node-canvas-webgl/lib')

const { WorldView, Viewer, Entity } = require('../viewer')
const { EventEmitter } = require('stream')

module.exports = (bot, { viewDistance = 6, width = 256, height = 144, jpegOptions, showSelfMesh = false }) => {
  const canvas = createCanvas(width, height)
  const renderer = new THREE.WebGLRenderer({ canvas })
  const viewer = new Viewer(renderer)

  viewer.setVersion(bot.version)
  viewer.setFirstPersonCamera(bot.entity.position, bot.entity.yaw, bot.entity.pitch)

  // Load world
  const worldView = new WorldView(bot.world, viewDistance, bot.entity.position)
  viewer.listen(worldView)
  worldView.init(bot.entity.position)
  worldView.listenToBot(bot)
  viewer.setFirstPersonCamera(bot.entity.position, bot.entity.yaw, bot.entity.pitch)

  // Bot self-mesh (for third-person / overhead views)
  let botMesh = null
  function addSelfMesh () {
    if (botMesh) return
    try {
      botMesh = new Entity('1.16.4', 'player', viewer.scene).mesh
      viewer.scene.add(botMesh)
      botMesh.position.set(bot.entity.position.x, bot.entity.position.y, bot.entity.position.z)
    } catch (e) {}
  }
  function updateSelfMesh () {
    if (!botMesh) return
    botMesh.position.set(bot.entity.position.x, bot.entity.position.y, bot.entity.position.z)
    botMesh.rotation.y = bot.entity.yaw
  }
  if (showSelfMesh) {
    addSelfMesh()
    bot.on('move', updateSelfMesh)
  }

  let freecamera = new EventEmitter()
  freecamera.set = ({ pos, yaw, pitch }) => {
    // Set camera position directly (skip TWEEN for immediate effect)
    viewer.camera.position.set(pos.x, pos.y + 1.6, pos.z)
    viewer.camera.rotation.set(pitch, yaw, 0, 'ZYX')
    worldView.updatePosition(pos)
  }
  freecamera.get = () => {
    TWEEN.update()
    viewer.update()
    renderer.render(viewer.scene, viewer.camera)
    return canvas.toBuffer('image/jpeg', jpegOptions).toString('base64');
  }
  freecamera.enableSelfMesh = () => {
    addSelfMesh()
    if (!showSelfMesh) {
      showSelfMesh = true
      bot.on('move', updateSelfMesh)
    }
  }

  let idx = 0
  function update () {
    viewer.update()
    renderer.render(viewer.scene, viewer.camera)
  }
  update()

  return freecamera
}
