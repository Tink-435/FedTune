# VIRTUAL-LAB ⚛️

A real-time collaborative 2D physics sandbox built for university-level STEM education. Multiple users can build, run, and analyse mechanical simulations together in a shared workspace — think Google Docs, but for physics experiments.

---

## The Problem

Teaching physics online is frustrating. Students watch static videos, read equations off a screen, and never actually *feel* how a pendulum behaves differently when you change its mass, or what happens to coupled springs when you crank up the stiffness. The gap between theory and intuition is hard to bridge without hands-on experimentation.

VIRTUAL-LAB tries to fix that by giving instructors and students a shared, live physics canvas where they can build experiments together and watch the math play out in real time.

---

## Demo

> 📹 **[Watch Demo Video](#)**
> *(replace the # above with your actual video link — YouTube, Loom, or Google Drive all work)*

---

## What It Does

### For Students
- Join a live room using a 6-character code from your instructor
- Add physics bodies (boxes, circles, triangles, hexagons) to the shared canvas
- Watch velocity and acceleration vectors update in real time on each body
- See energy conservation graphs — kinetic energy, potential energy, and total energy plotted live
- Save experiments to your personal library and revisit them later
- Browse the experiment library and clone templates to explore on your own

### For Instructors
- Create a room and share the join code with your class
- Load preset experiments: Free Fall, Simple Harmonic Motion, Coupled Springs, Pendulum, Collision
- Lock the room so students can only observe while you demonstrate
- Trigger chaos events that broadcast to every student simultaneously:
  - **Flip Gravity** — reverses gravitational direction
  - **Shockwave** — applies an outward impulse from the canvas centre
  - **Freeze All** — stops every body mid-motion (great for pausing a demo to discuss)
  - **Zero Gravity** — removes gravity entirely
  - **Pin Mode** — click any body to anchor it in place
- Publish experiment templates to the library with instructions and a grading rubric
- Review and grade student submissions

---

## How Real-Time Sync Works

This was the hardest part to get right. When the instructor drags a body, every student sees it move. When anyone adds a spring pair, it appears on all canvases simultaneously.

Under the hood, every canvas action (add body, load preset, drag, reset) emits a `canvas_action` socket event. The server rebroadcasts it to everyone else in the room, and each client applies the change to their local Matter.js world.

For drag sync specifically, position updates fire on every `mousemove` while dragging. The server skips the database entirely for these and just rebroadcasts instantly — otherwise at 60fps you would be firing thousands of DB queries per second.

---

## Tech Stack

| Layer | Tech | Why |
|---|---|---|
| Frontend | React 18 | Component model, hooks for state |
| Physics | Matter.js | Best 2D rigid body engine for browsers |
| Real-time | Socket.io | WebSocket with fallback, rooms built-in |
| Charts | Recharts | React-native, handles streaming data cleanly |
| Backend | Node.js + Express | Non-blocking I/O, good fit for socket-heavy servers |
| Database | MongoDB + Mongoose | Flexible schema for physics state JSON |
| Auth | JWT + bcrypt | Stateless, horizontally scalable |
| DevOps | Docker + GitHub Actions | One-command setup, automated CI |

---

## Architecture

```
Browser (React)
  │
  ├── AuthContext     → REST calls via axios (login, register, experiments)
  ├── SocketContext   → Socket.io connection (one per session)
  │
  └── PhysicsCanvas
        ├── Matter.js engine  → runs physics simulation
        ├── canvas_action     → emits on every user interaction
        └── Recharts          → plots KE, PE, velocity live

        ↕  WebSocket

Node.js Server
  ├── Express REST API
  │     /api/auth         → JWT register / login
  │     /api/rooms        → create, join, lock, state persistence
  │     /api/experiments  → save, publish, clone, submit, grade
  │
  └── Socket.io Manager
        join_room       → sends current room state to new joiner
        canvas_action   → rebroadcasts to all other users in room
        chaos_event     → instructor-only, broadcasts to everyone
        toggle_lock     → locks/unlocks room for students

        ↕  Mongoose ODM

MongoDB
  ├── Users       → bcrypt passwords, RBAC roles
  ├── Rooms       → physics state snapshots, participant list, TTL expiry
  └── Experiments → versioned saves, submissions, grades
```

---

## Running Locally

### Option A — Docker (recommended)

Make sure Docker Desktop is open and running first (look for the whale icon in your menu bar).

```bash
# 1. Clone the repo
# ⚠️  REPLACE with your actual GitHub repo URL below
git clone https://github.com/YOUR_USERNAME/virtual-lab
cd virtual-lab

# 2. Set up environment variables
cp backend/.env.example backend/.env
# Open backend/.env in any text editor and set JWT_SECRET to any random string
# Example: JWT_SECRET=myrandombsecretkey123456789

# 3. Start everything
docker-compose up --build
```

Once running:
- Frontend → http://localhost:3001
- Backend API → http://localhost:5001
- Health check → http://localhost:5001/health

To stop: press `Ctrl+C` then run `docker-compose down`

---

### Option B — Manual (no Docker)

You will need Node.js 18+ and MongoDB installed locally.

**Terminal 1 — Start MongoDB**
```bash
mongod
```

**Terminal 2 — Start Backend**
```bash
cd backend
cp .env.example .env
npm install
npm run dev
```

**Terminal 3 — Start Frontend**
```bash
cd frontend
npm install
npm start
```

Frontend opens at http://localhost:3000, backend runs at http://localhost:5000.

---

### Testing It End to End

1. Go to `http://localhost:3001`
2. Register an **Instructor** account
3. Open an incognito window and register a **Student** account
4. Instructor: click **Create Room** and copy the 6-character code
5. Student: paste the code into **Join a Room**
6. Both users are now live in the same physics canvas

Try loading the **Pendulum** preset on the instructor side — the student should see it appear instantly. Then try **Flip Gravity** from the instructor's chaos panel.

---

## Running Tests

```bash
cd backend
npm test
npm test -- --coverage
```

Tests cover the OT conflict resolution engine (unit tests) and auth routes (integration tests against an in-memory MongoDB instance).

---

## Project Structure

```
virtual-lab/
├── .github/workflows/ci.yml      → GitHub Actions CI pipeline
├── docker-compose.yml             → one-command full stack setup
│
├── backend/
│   ├── src/
│   │   ├── server.js              → Express + Socket.io entry point
│   │   ├── config/db.js           → MongoDB connection
│   │   ├── middleware/auth.js     → JWT protect + RBAC authorize
│   │   ├── models/                → User, Room, Experiment schemas
│   │   ├── controllers/           → business logic
│   │   ├── routes/                → Express routers
│   │   └── socket/
│   │       ├── socketManager.js   → all real-time event handling
│   │       └── otEngine.js        → conflict resolution engine
│   └── Dockerfile
│
└── frontend/
    ├── src/
    │   ├── context/
    │   │   ├── AuthContext.jsx    → global auth state + axios instance
    │   │   └── SocketContext.jsx  → socket.io connection management
    │   ├── components/
    │   │   ├── Canvas/
    │   │   │   └── PhysicsCanvas.jsx  → main lab workspace
    │   │   ├── Auth/AuthPage.jsx
    │   │   └── Library/ExperimentLibrary.jsx
    │   └── pages/
    │       ├── Dashboard.jsx
    │       ├── RoomPage.jsx
    │       └── MyExperiments.jsx
    └── Dockerfile
```

---

## What I Learned Building This

**Real-time sync is harder than it looks.** The trickiest bug in this project was that the Socket.io client connected asynchronously, but the React component was reading `socketRef.current` (which was `null` at render time) instead of waiting for the connection. The fix was switching from a ref to actual React state for the socket instance, so components re-render when the connection is ready.

**Physics engines are opinionated.** Matter.js runs on a fixed timestep and does not care about your React component lifecycle. Learning to separate the simulation loop from the render loop — and not letting them block each other — was a big part of getting smooth 60fps with live analytics at the same time.

**Drag sync needs special handling.** Syncing discrete events like adding a body or loading a preset is straightforward. Syncing continuous mouse drag at 60fps is different — you cannot hit the database on every frame. Skipping the DB lookup for move events and just rebroadcasting them inline made drag feel instant.

---

## Known Limitations and Future Work

- [ ] If two users drag the same body simultaneously, the last update wins — no full conflict resolution for drag events yet
- [ ] Physics state is not perfectly deterministic across clients — floating point differences can cause tiny divergence over time
- [ ] No mobile or touch support on the canvas yet
- [ ] Would love to add a session replay system — record a full experiment and play it back later

---

## Screenshots

> ⚠️  Add your screenshots here. Recommended: one of the login page, one of the canvas with bodies and vectors, one of the analytics panel.
>
> To add a screenshot in markdown:
> ![Description of screenshot](./screenshots/your-image-name.png)
>
> Create a /screenshots folder in the root of the project and drop your images there.

---

*Built with Matter.js · Socket.io · React · Node.js · MongoDB*
