import streamlit as st
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Dot Connection Background", layout="wide")

# Define the HTML and JavaScript for the background effect
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      overflow: hidden;
      background-color: #0d1117;
    }
    #dotCanvas {
      position: fixed;
      top: 0;
      left: 0;
      z-index: -1;
    }
  </style>
</head>
<body>
  <canvas id="dotCanvas"></canvas>
  <script>
    const canvas = document.getElementById('dotCanvas');
    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];
    const particleCount = 100;
    const maxDistance = 120;
    let mouse = { x: null, y: null };

    function resizeCanvas() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 1.5,
        vy: (Math.random() - 0.5) * 1.5,
      });
    }

    window.addEventListener('mousemove', function(e) {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    });

    function animate() {
      ctx.clearRect(0, 0, width, height);
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = '#58a6ff';
        ctx.fill();
      });

      for (let i = 0; i < particleCount; i++) {
        for (let j = i + 1; j < particleCount; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < maxDistance) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = 'rgba(88, 166, 255, 0.1)';
            ctx.stroke();
          }
        }
      }

      if (mouse.x && mouse.y) {
        particles.forEach(p => {
          const dx = p.x - mouse.x;
          const dy = p.y - mouse.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < maxDistance) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(mouse.x, mouse.y);
            ctx.strokeStyle = 'rgba(88, 166, 255, 0.2)';
            ctx.stroke();
          }
        });
      }

      requestAnimationFrame(animate);
    }

    animate();
  </script>
</body>
</html>
"""

# Embed the HTML and JavaScript into the Streamlit app
components.html(html_code, height=0, width=0)

# Your main Streamlit content goes here
st.title("🔗 Dot Connection Effect as Background")
st.write("This Streamlit app features a dynamic dot connection effect as the full-page background.")
