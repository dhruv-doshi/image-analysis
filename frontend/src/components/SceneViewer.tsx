'use client'

/**
 * SceneViewer — Interactive 3-D point-cloud built from a monocular depth map.
 *
 * Pipeline:
 *  1. Decode the base64 depth PNG + colour JPEG from the /depth API response.
 *  2. Draw both onto off-screen <canvas> elements to read per-pixel RGBA values.
 *  3. Build a Three.js BufferGeometry where each pixel becomes a coloured vertex:
 *       x = normalised horizontal position  (-aspect … +aspect)
 *       y = normalised vertical position    (-1 … +1)
 *       z = depth pixel value              (0 = far, 1 = close to camera)
 *  4. Render with OrbitControls so the user can drag, zoom, and pan to find their
 *     ideal shooting angle.
 *  5. Overlay a Rule-of-Thirds grid as thin line segments in 3-D space.
 */

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { DepthResult } from '@/types/api'

interface Props {
  depthResult: DepthResult
}

// ─── helpers ────────────────────────────────────────────────────────────────

/** Decode a base64 data string into pixel RGBA via an off-screen canvas. */
function decodeBase64ToPixels(
  b64: string,
  mimeType: 'image/png' | 'image/jpeg',
): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) return reject(new Error('Could not get 2D context'))
      ctx.drawImage(img, 0, 0)
      const { data } = ctx.getImageData(0, 0, img.naturalWidth, img.naturalHeight)
      resolve({ data, width: img.naturalWidth, height: img.naturalHeight })
    }
    img.onerror = () => reject(new Error('Image decode failed'))
    img.src = `data:${mimeType};base64,${b64}`
  })
}

/** Add a Rule-of-Thirds grid as 3-D line segments at the given z depth. */
function addRoTGrid(scene: THREE.Scene, aspect: number, z: number) {
  const mat = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.25,
  })

  const thirds = [1 / 3, 2 / 3]

  // Vertical lines
  for (const t of thirds) {
    const x = (t * 2 - 1) * aspect
    const pts = [new THREE.Vector3(x, -1, z), new THREE.Vector3(x, 1, z)]
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat))
  }

  // Horizontal lines
  for (const t of thirds) {
    const y = -(t * 2 - 1)
    const pts = [new THREE.Vector3(-aspect, y, z), new THREE.Vector3(aspect, y, z)]
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat))
  }
}

// ─── component ──────────────────────────────────────────────────────────────

export default function SceneViewer({ depthResult }: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    let animId = 0
    let renderer: THREE.WebGLRenderer | null = null

    ;(async () => {
      try {
        // 1. Decode images ------------------------------------------------
        const [colorData, depthData] = await Promise.all([
          decodeBase64ToPixels(depthResult.image, 'image/jpeg'),
          decodeBase64ToPixels(depthResult.depth_map, 'image/png'),
        ])

        const { width: W, height: H } = colorData
        const aspect = W / H
        const depthScale = 0.6 // how far points pop out along z

        // 2. Build point-cloud geometry -----------------------------------
        const N = W * H
        const positions = new Float32Array(N * 3)
        const colors = new Float32Array(N * 3)

        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const idx = y * W + x

            // Normalised screen-space x/y
            const nx = ((x / (W - 1)) * 2 - 1) * aspect
            const ny = -((y / (H - 1)) * 2 - 1)

            // Depth from grayscale (R channel of RGBA)
            const dIdx = idx * 4
            const dv = depthData.data[dIdx] / 255 // 0=far, 1=close

            positions[idx * 3 + 0] = nx
            positions[idx * 3 + 1] = ny
            positions[idx * 3 + 2] = dv * depthScale

            // Colour (RGB)
            const cIdx = idx * 4
            colors[idx * 3 + 0] = colorData.data[cIdx + 0] / 255
            colors[idx * 3 + 1] = colorData.data[cIdx + 1] / 255
            colors[idx * 3 + 2] = colorData.data[cIdx + 2] / 255
          }
        }

        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))

        const mat = new THREE.PointsMaterial({
          size: 0.004,
          vertexColors: true,
          sizeAttenuation: true,
        })
        const points = new THREE.Points(geo, mat)

        // 3. Scene setup --------------------------------------------------
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0x09090b) // zinc-950

        scene.add(points)
        addRoTGrid(scene, aspect, depthScale * 0.5) // grid at mid depth

        const cw = mount.clientWidth
        const ch = mount.clientHeight

        const camera = new THREE.PerspectiveCamera(55, cw / ch, 0.01, 50)
        camera.position.set(0, 0, 2.2)

        renderer = new THREE.WebGLRenderer({ antialias: true })
        renderer.setSize(cw, ch)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        mount.appendChild(renderer.domElement)

        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.06
        controls.minDistance = 0.5
        controls.maxDistance = 5

        // Resize observer
        const ro = new ResizeObserver(() => {
          const w = mount.clientWidth
          const h = mount.clientHeight
          camera.aspect = w / h
          camera.updateProjectionMatrix()
          renderer?.setSize(w, h)
        })
        ro.observe(mount)

        function animate() {
          animId = requestAnimationFrame(animate)
          controls.update()
          renderer!.render(scene, camera)
        }
        animate()

        setStatus('ready')

        // Cleanup
        return () => {
          ro.disconnect()
          cancelAnimationFrame(animId)
          controls.dispose()
          geo.dispose()
          mat.dispose()
          renderer?.dispose()
          if (mount.contains(renderer?.domElement ?? null)) {
            mount.removeChild(renderer!.domElement)
          }
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : 'Failed to build 3D scene')
        setStatus('error')
      }
    })()

    return () => {
      cancelAnimationFrame(animId)
      renderer?.dispose()
    }
  }, [depthResult])

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-zinc-500">
          Drag to orbit &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Right-drag to pan
        </p>
        {status === 'loading' && (
          <span className="text-xs text-indigo-400 flex items-center gap-1.5">
            <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Building scene…
          </span>
        )}
      </div>

      {status === 'error' ? (
        <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {errorMsg}
        </div>
      ) : (
        <div
          ref={mountRef}
          className="w-full rounded-xl overflow-hidden border border-zinc-800"
          style={{ height: 420 }}
        />
      )}

      <p className="text-[11px] text-zinc-600">
        White grid = Rule of Thirds overlay &nbsp;·&nbsp; Depth powered by Depth Anything V2
      </p>
    </div>
  )
}
