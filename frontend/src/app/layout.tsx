import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aegis DFI — AI-Powered Drug-Food Interaction Platform",
  description: "Predict and explain drug-food interaction risks with XGBoost classification, RDKit molecular descriptors, and SHAP explainable AI insights.",
  keywords: ["drug-food interaction", "pharmacology", "machine learning", "bioinformatics", "explainable AI", "RDKit", "XGBoost", "SHAP"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col antialiased bg-[#070504]">
        {children}
      </body>
    </html>
  );
}
