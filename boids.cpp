// boids.cpp
// Boids flocking simulation using SDL2, built to run in the browser with Emscripten.
//
// Controls:
//   Hold left mouse button (or touch) to attract the flock.
//   Hold right mouse button to scare the flock away.
//   Sliders in the page (see shell.html) call the exported set_* functions below.

#include <SDL2/SDL.h>
#include <emscripten.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

// Canvas size in pixels (the page scales it with CSS)
constexpr int kCanvasWidth = 900;
constexpr int kCanvasHeight = 600;

constexpr float kPi = 3.14159265f;
constexpr int kMaxBoidCount = 800;

// Minimal 2D vector used for positions, velocities and steering forces
struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

// One member of the flock
struct Boid {
    Vec2 position;
    Vec2 velocity;
};

// Tunable values, some of which are controlled by the page sliders
struct Settings {
    float separationWeight = 1.6f;   // how strongly boids avoid crowding
    float alignmentWeight = 1.0f;    // how strongly boids match heading
    float cohesionWeight = 0.9f;     // how strongly boids drift to the group centre
    float perceptionRadius = 55.0f;  // how far a boid can "see" neighbours
    float separationRadius = 22.0f;  // distance at which boids start to push apart
    float maxSpeed = 3.2f;           // top speed in pixels per frame
    float minSpeed = 1.4f;           // keeps the flock from stalling
    float maxSteerForce = 0.07f;     // limit on how sharply a boid can turn
    float mouseRadius = 260.0f;      // reach of the attract and scare effect
    float maxMouseSteer = 0.35f;     // strongest turn the cursor can force on a boid per frame
    float mouseArriveRadius = 70.0f; // boids slow down inside this distance so they gather at the cursor
};

// Global state, needed because the Emscripten main loop takes a plain function
SDL_Window* gWindow = nullptr;
SDL_Renderer* gRenderer = nullptr;
std::vector<Boid> gBoids;
Settings gSettings;
std::mt19937 gRandomEngine{std::random_device{}()};

float Length(const Vec2& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

// Shrinks a vector so its length never exceeds maxLength
Vec2 Limit(const Vec2& v, float maxLength)
{
    const float length = Length(v);
    if (length > maxLength && length > 0.0f) {
        const float scale = maxLength / length;
        return {v.x * scale, v.y * scale};
    }
    return v;
}

// Rescales a vector to an exact length (returns zero vector if it has no direction)
Vec2 SetLength(const Vec2& v, float newLength)
{
    const float length = Length(v);
    if (length <= 0.0f) {
        return {0.0f, 0.0f};
    }
    const float scale = newLength / length;
    return {v.x * scale, v.y * scale};
}

// Turns a desired direction into a steering force the boid is allowed to apply
Vec2 SteerToward(const Vec2& desiredDirection, const Vec2& currentVelocity)
{
    const Vec2 desiredVelocity = SetLength(desiredDirection, gSettings.maxSpeed);
    const Vec2 steering = {
        desiredVelocity.x - currentVelocity.x,
        desiredVelocity.y - currentVelocity.y,
    };
    return Limit(steering, gSettings.maxSteerForce);
}

float RandomRange(float low, float high)
{
    std::uniform_real_distribution<float> distribution(low, high);
    return distribution(gRandomEngine);
}

// Creates a boid at a random spot moving in a random direction
Boid MakeRandomBoid()
{
    Boid boid;
    boid.position = {RandomRange(0.0f, kCanvasWidth), RandomRange(0.0f, kCanvasHeight)};
    const float heading = RandomRange(0.0f, 2.0f * kPi);
    boid.velocity = {std::cos(heading) * gSettings.maxSpeed, std::sin(heading) * gSettings.maxSpeed};
    return boid;
}

// Converts hue, saturation and value (all 0 to 1) into an SDL colour
SDL_Color HsvToColor(float hue, float saturation, float value)
{
    const float scaledHue = hue * 6.0f;
    const int sector = static_cast<int>(scaledHue) % 6;
    const float fraction = scaledHue - std::floor(scaledHue);
    const float p = value * (1.0f - saturation);
    const float q = value * (1.0f - fraction * saturation);
    const float t = value * (1.0f - (1.0f - fraction) * saturation);

    float red = value;
    float green = t;
    float blue = p;
    switch (sector) {
        case 0: red = value; green = t; blue = p; break;
        case 1: red = q; green = value; blue = p; break;
        case 2: red = p; green = value; blue = t; break;
        case 3: red = p; green = q; blue = value; break;
        case 4: red = t; green = p; blue = value; break;
        default: red = value; green = p; blue = q; break;
    }

    return {
        static_cast<Uint8>(red * 255.0f),
        static_cast<Uint8>(green * 255.0f),
        static_cast<Uint8>(blue * 255.0f),
        255,
    };
}

// Advances every boid by one frame
void UpdateFlock(bool attractPressed, bool scarePressed, const Vec2& mousePosition)
{
    const float perceptionSquared = gSettings.perceptionRadius * gSettings.perceptionRadius;
    const float separationSquared = gSettings.separationRadius * gSettings.separationRadius;
    const float halfWidth = kCanvasWidth * 0.5f;
    const float halfHeight = kCanvasHeight * 0.5f;

    // Compute every new velocity from the current positions first, then apply them,
    // so the order the boids are processed in does not affect the result
    std::vector<Vec2> newVelocities(gBoids.size());

    for (std::size_t index = 0; index < gBoids.size(); ++index) {
        const Boid& current = gBoids[index];

        Vec2 separationSum;   // pushes away from very close neighbours
        Vec2 alignmentSum;    // sum of neighbour velocities
        Vec2 cohesionOffset;  // sum of offsets pointing to neighbours
        int neighbourCount = 0;

        for (std::size_t otherIndex = 0; otherIndex < gBoids.size(); ++otherIndex) {
            if (otherIndex == index) {
                continue;
            }
            const Boid& other = gBoids[otherIndex];

            // Shortest offset to the other boid, accounting for screen wrap-around
            float offsetX = other.position.x - current.position.x;
            float offsetY = other.position.y - current.position.y;
            if (offsetX > halfWidth) { offsetX -= kCanvasWidth; }
            if (offsetX < -halfWidth) { offsetX += kCanvasWidth; }
            if (offsetY > halfHeight) { offsetY -= kCanvasHeight; }
            if (offsetY < -halfHeight) { offsetY += kCanvasHeight; }

            const float distanceSquared = offsetX * offsetX + offsetY * offsetY;
            if (distanceSquared > perceptionSquared || distanceSquared <= 0.0f) {
                continue;
            }

            ++neighbourCount;
            alignmentSum.x += other.velocity.x;
            alignmentSum.y += other.velocity.y;
            cohesionOffset.x += offsetX;
            cohesionOffset.y += offsetY;

            // Closer neighbours push harder (weight is 1 / distance)
            if (distanceSquared < separationSquared) {
                const float distance = std::sqrt(distanceSquared);
                separationSum.x -= offsetX / distance / distance;
                separationSum.y -= offsetY / distance / distance;
            }
        }

        Vec2 acceleration;

        if (neighbourCount > 0) {
            // Alignment: steer toward the average heading of neighbours
            const Vec2 alignSteer = SteerToward(alignmentSum, current.velocity);
            // Cohesion: steer toward the average position of neighbours
            const Vec2 cohesionSteer = SteerToward(cohesionOffset, current.velocity);

            acceleration.x += alignSteer.x * gSettings.alignmentWeight;
            acceleration.y += alignSteer.y * gSettings.alignmentWeight;
            acceleration.x += cohesionSteer.x * gSettings.cohesionWeight;
            acceleration.y += cohesionSteer.y * gSettings.cohesionWeight;
        }

        // Separation: steer away from crowding neighbours
        if (Length(separationSum) > 0.0f) {
            const Vec2 separationSteer = SteerToward(separationSum, current.velocity);
            acceleration.x += separationSteer.x * gSettings.separationWeight;
            acceleration.y += separationSteer.y * gSettings.separationWeight;
        }

        // Mouse interaction: left button attracts, right button scares
        if (attractPressed || scarePressed) {
            const float toMouseX = mousePosition.x - current.position.x;
            const float toMouseY = mousePosition.y - current.position.y;
            const float mouseDistance = std::sqrt(toMouseX * toMouseX + toMouseY * toMouseY);
            if (mouseDistance > 0.0f && mouseDistance < gSettings.mouseRadius) {
                // Influence is strongest near the cursor and fades toward the edge of the radius
                const float influence = 1.0f - mouseDistance / gSettings.mouseRadius;
                const float steerLimit = gSettings.maxMouseSteer * (0.35f + 0.65f * influence);

                Vec2 desiredVelocity;
                if (attractPressed) {
                    // Seek: head for the cursor, slowing down as it gets close
                    // so the flock gathers around it instead of flying straight past
                    const float arriveFactor = std::min(1.0f, mouseDistance / gSettings.mouseArriveRadius);
                    desiredVelocity = SetLength({toMouseX, toMouseY}, gSettings.maxSpeed * arriveFactor);
                } else {
                    // Flee: head directly away from the cursor at full speed
                    desiredVelocity = SetLength({-toMouseX, -toMouseY}, gSettings.maxSpeed);
                }

                // Steering is the gap between the wanted velocity and the current one, capped by steerLimit
                const Vec2 mouseSteer = Limit(
                    {desiredVelocity.x - current.velocity.x, desiredVelocity.y - current.velocity.y},
                    steerLimit
                );
                acceleration.x += mouseSteer.x;
                acceleration.y += mouseSteer.y;
            }
        }

        Vec2 velocity = {
            current.velocity.x + acceleration.x,
            current.velocity.y + acceleration.y,
        };
        velocity = Limit(velocity, gSettings.maxSpeed);

        // Keep every boid moving so the flock never freezes in place
        if (Length(velocity) < gSettings.minSpeed) {
            velocity = SetLength(velocity, gSettings.minSpeed);
        }
        newVelocities[index] = velocity;
    }

    // Apply the new velocities and wrap boids around the screen edges
    for (std::size_t index = 0; index < gBoids.size(); ++index) {
        Boid& boid = gBoids[index];
        boid.velocity = newVelocities[index];
        boid.position.x += boid.velocity.x;
        boid.position.y += boid.velocity.y;

        if (boid.position.x < 0.0f) { boid.position.x += kCanvasWidth; }
        if (boid.position.x >= kCanvasWidth) { boid.position.x -= kCanvasWidth; }
        if (boid.position.y < 0.0f) { boid.position.y += kCanvasHeight; }
        if (boid.position.y >= kCanvasHeight) { boid.position.y -= kCanvasHeight; }
    }
}

// Draws every boid as a small triangle pointing along its heading,
// coloured by direction so you can see the flock turn together
void DrawFlock()
{
    constexpr float kBoidLength = 11.0f;  // tip to tail
    constexpr float kBoidHalfWidth = 4.5f;

    std::vector<SDL_Vertex> vertices;
    vertices.reserve(gBoids.size() * 3);

    for (const Boid& boid : gBoids) {
        const float heading = std::atan2(boid.velocity.y, boid.velocity.x);

        // Direction the boid faces, and the direction at a right angle to it
        const float forwardX = std::cos(heading);
        const float forwardY = std::sin(heading);
        const float sideX = -forwardY;
        const float sideY = forwardX;

        // Map heading (-pi to pi) onto a teal-to-pink hue range
        const float headingFraction = (heading + kPi) / (2.0f * kPi);
        const float hue = 0.48f + 0.5f * headingFraction;
        const SDL_Color color = HsvToColor(std::fmod(hue, 1.0f), 0.45f, 0.96f);

        const SDL_FPoint tip = {
            boid.position.x + forwardX * kBoidLength * 0.6f,
            boid.position.y + forwardY * kBoidLength * 0.6f,
        };
        const SDL_FPoint leftTail = {
            boid.position.x - forwardX * kBoidLength * 0.4f + sideX * kBoidHalfWidth,
            boid.position.y - forwardY * kBoidLength * 0.4f + sideY * kBoidHalfWidth,
        };
        const SDL_FPoint rightTail = {
            boid.position.x - forwardX * kBoidLength * 0.4f - sideX * kBoidHalfWidth,
            boid.position.y - forwardY * kBoidLength * 0.4f - sideY * kBoidHalfWidth,
        };

        vertices.push_back({tip, color, {0.0f, 0.0f}});
        vertices.push_back({leftTail, color, {0.0f, 0.0f}});
        vertices.push_back({rightTail, color, {0.0f, 0.0f}});
    }

    // One batched draw call for the whole flock keeps it fast
    SDL_RenderGeometry(gRenderer, nullptr, vertices.data(), static_cast<int>(vertices.size()), nullptr, 0);
}

// Draws a soft ring around the cursor while a mouse button is held,
// showing how far the attract or scare effect reaches
void DrawMouseRing(const Vec2& mousePosition, bool attractPressed)
{
    constexpr int kRingSegments = 64;
    SDL_FPoint ringPoints[kRingSegments + 1];

    // Walk around a circle, closing the loop by repeating the first point at the end
    for (int segment = 0; segment <= kRingSegments; ++segment) {
        const float angle = 2.0f * kPi * static_cast<float>(segment) / static_cast<float>(kRingSegments);
        ringPoints[segment] = {
            mousePosition.x + std::cos(angle) * gSettings.mouseRadius,
            mousePosition.y + std::sin(angle) * gSettings.mouseRadius,
        };
    }

    SDL_SetRenderDrawBlendMode(gRenderer, SDL_BLENDMODE_BLEND);
    if (attractPressed) {
        SDL_SetRenderDrawColor(gRenderer, 240, 150, 190, 110);  // pink ring when attracting
    } else {
        SDL_SetRenderDrawColor(gRenderer, 120, 210, 200, 110);  // teal ring when scaring
    }
    SDL_RenderDrawLinesF(gRenderer, ringPoints, kRingSegments + 1);
    SDL_SetRenderDrawBlendMode(gRenderer, SDL_BLENDMODE_NONE);
}

// Runs once per browser animation frame
void RunFrame()
{
    // Drain the event queue so SDL keeps its mouse and touch state up to date
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
    }

    int mouseX = 0;
    int mouseY = 0;
    const Uint32 mouseButtons = SDL_GetMouseState(&mouseX, &mouseY);
    const bool attractPressed = (mouseButtons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    const bool scarePressed = (mouseButtons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
    const Vec2 mousePosition = {static_cast<float>(mouseX), static_cast<float>(mouseY)};

    UpdateFlock(attractPressed, scarePressed, mousePosition);

    // Soft dark background so the pastel boids stand out
    SDL_SetRenderDrawColor(gRenderer, 22, 26, 40, 255);
    SDL_RenderClear(gRenderer);
    DrawFlock();
    if (attractPressed || scarePressed) {
        DrawMouseRing(mousePosition, attractPressed);
    }
    SDL_RenderPresent(gRenderer);
}

}  // namespace

// Functions below are called from JavaScript (the sliders in shell.html)
extern "C" {

EMSCRIPTEN_KEEPALIVE void set_separation_weight(float value)
{
    gSettings.separationWeight = value;
}

EMSCRIPTEN_KEEPALIVE void set_alignment_weight(float value)
{
    gSettings.alignmentWeight = value;
}

EMSCRIPTEN_KEEPALIVE void set_cohesion_weight(float value)
{
    gSettings.cohesionWeight = value;
}

EMSCRIPTEN_KEEPALIVE void set_boid_count(int requestedCount)
{
    const std::size_t newCount = static_cast<std::size_t>(std::clamp(requestedCount, 1, kMaxBoidCount));

    // Add new random boids when growing, drop the extras when shrinking
    while (gBoids.size() < newCount) {
        gBoids.push_back(MakeRandomBoid());
    }
    gBoids.resize(newCount);
}

}  // extern "C"

int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    gWindow = SDL_CreateWindow(
        "Boids",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        kCanvasWidth,
        kCanvasHeight,
        SDL_WINDOW_SHOWN
    );
    gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    // Start with a medium sized flock
    set_boid_count(250);

    // 0 fps tells Emscripten to use the browser's requestAnimationFrame timing
    emscripten_set_main_loop(RunFrame, 0, 1);

    return 0;
}
