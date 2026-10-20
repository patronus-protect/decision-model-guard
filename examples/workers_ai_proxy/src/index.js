export default {
  async fetch(req, env) {
    const m = new URL(req.url).pathname.match(/^\/run\/(.+)$/);
    if (req.method !== "POST" || !m) return new Response("POST /run/<model>", { status: 404 });
    try {
      const result = await env.AI.run(decodeURIComponent(m[1]), await req.json());
      return Response.json({ result, success: true });
    } catch (e) {
      return Response.json({ success: false, errors: [{ message: String((e && e.message) || e) }] }, { status: 502 });
    }
  },
};
