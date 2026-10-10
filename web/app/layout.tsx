import type { ReactNode } from "react";

export const metadata = {
  title: "Job Application Copilot",
  description: "Match, tailor and prep for jobs in India — with you approving every change.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en-IN">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, background: "#fafaf9" }}>
        <main style={{ maxWidth: 880, margin: "0 auto", padding: "24px 16px" }}>{children}</main>
      </body>
    </html>
  );
}
