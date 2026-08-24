// Drive the `visible` flag of named models inside an embedded Bokeh document --
// how a control flips between pre-built variants of a figure (e.g. the Loss tab's
// linear/log x-axis rows) without a round trip to the API or a re-embed. Kept
// free of BokehJS imports so it is unit-testable against a stub document.

// The slice of BokehJS's Document / model API this relies on.
export interface NamedModel { visible: boolean }
export interface NamedModelDocument {
  get_model_by_name(name: string): NamedModel | null;
}

// Model name -> whether it should be displayed.
export type Visibility = Record<string, boolean>;

export function applyVisibility(doc: NamedModelDocument | null | undefined, visibility: Visibility) {
  if (!doc) return;
  for (const [name, visible] of Object.entries(visibility)) {
    const model = doc.get_model_by_name(name);
    if (model && model.visible !== visible) model.visible = visible;
  }
}
