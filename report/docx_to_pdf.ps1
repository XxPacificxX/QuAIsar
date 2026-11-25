<#
.SYNOPSIS
    Convert a .docx to PDF with Microsoft Word (Windows). Used by build_report.py --pdf.
.EXAMPLE
    .\report\docx_to_pdf.ps1 report\QuAIsar-findings.docx
#>
param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [string]$Pdf
)
$ErrorActionPreference = "Stop"
$Docx = (Resolve-Path $Docx).Path
if (-not $Pdf) { $Pdf = [IO.Path]::ChangeExtension($Docx, ".pdf") }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $true)  # ConfirmConversions, ReadOnly
    $doc.Fields.Update() | Out-Null                       # page numbers, page count
    $doc.SaveAs2($Pdf, 17)                                 # 17 = wdFormatPDF
    $doc.Close($false)
    Write-Host "Wrote $Pdf"
}
finally {
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
